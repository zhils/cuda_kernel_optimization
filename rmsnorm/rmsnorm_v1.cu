// RMSNorm V1: 2维 block + float4 协作加载 + 共享内存 staging
//
// 核心设计:
// — 2维 block: br 行 × bc 列线程, 每个 block 同时处理 br 行数据
// — float4 向量化: 每个线程每次加载 4 个 float, 全局内存带宽最大化
// — 共享内存 staging: 整行数据先加载到 smem, 计算和写回都从 smem 进行
// — 串行归约: 每行第一个线程串行计算 sq_sum

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"
#include "rmsnorm/test_utils.h"

#define EPS 1e-5f
#define BR 8
#define BC 32

// ---------- 2D Block Kernel: 协作加载 + 共享内存计算 ----------
// 每个 block 处理 BR 行, 每行由 BC 个线程协作
// 共享内存布局: [BR][cols + padding] 存储输入数据
__global__ void RMSNormV1Kernel(const float* __restrict__ x,
                                float* __restrict__ y,
                                const float* __restrict__ weight,
                                int rows, int cols, float eps) {
    // 共享内存: 存储 BR 行数据, 每行 +1 padding 避免 bank conflict
    extern __shared__ float smem[];
    float* sdata = smem;  // [BR][cols + 1]

    const int tx = threadIdx.x;  // 列方向线程 id [0, BC)
    const int ty = threadIdx.y;  // 行方向线程 id [0, BR)
    const int row_base = blockIdx.x * BR;  // block 负责的起始行
    const int row = row_base + ty;          // 当前线程负责的行

    if (row >= rows) return;

    const float* row_x = x + static_cast<std::size_t>(row) * cols;
    float* row_y = y + static_cast<std::size_t>(row) * cols;
    float* s_row = sdata + ty * (cols + 1);  // 当前行在 smem 中的起始位置

    // ---- Step 1: 协作加载全局内存 → 共享内存 (float4) ----
    // 每行 BC 个线程协作, 每个线程加载 float4
    const int cols4 = cols / 4;
    for (int c = tx; c < cols4; c += BC) {
        const float4 v = *reinterpret_cast<const float4*>(row_x + c * 4);
        s_row[c * 4 + 0] = v.x;
        s_row[c * 4 + 1] = v.y;
        s_row[c * 4 + 2] = v.z;
        s_row[c * 4 + 3] = v.w;
    }
    // 处理尾部 (cols 不是 4 的倍数时)
    for (int c = cols4 * 4 + tx; c < cols; c += BC) {
        s_row[c] = row_x[c];
    }
    __syncthreads();

    // ---- Step 2: 串行计算平方和 (仅每行第一个线程执行) ----
    if (tx == 0) {
        float sq_sum = 0.f;
        const int c4 = cols / 4;
        for (int c = 0; c < c4; ++c) {
            const float4 v = *reinterpret_cast<const float4*>(s_row + c * 4);
            sq_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        for (int c = c4 * 4; c < cols; ++c) {
            const float v = s_row[c];
            sq_sum += v * v;
        }
        s_row[cols] = sq_sum;  // 暂存到行尾 (预留的 cols+1 空间)
    }
    __syncthreads();

    // 计算 RMS
    const float sq_sum = s_row[cols];
    const float rms = rsqrtf(sq_sum / static_cast<float>(cols) + eps);

    // ---- Step 3: 写回结果 (共享内存 → 全局内存, float4) ----
    for (int c = tx; c < cols4; c += BC) {
        const float4 vx = *reinterpret_cast<const float4*>(s_row + c * 4);
        const float4 vw = *reinterpret_cast<const float4*>(weight + c * 4);
        *reinterpret_cast<float4*>(row_y + c * 4) =
            make_float4(vx.x * rms * vw.x, vx.y * rms * vw.y,
                        vx.z * rms * vw.z, vx.w * rms * vw.w);
    }
    for (int c = cols4 * 4 + tx; c < cols; c += BC) {
        row_y[c] = s_row[c] * rms * weight[c];
    }
}

static void RMSNormCPU(const float* x, float* y, const float* weight, int rows, int cols,
                       float eps) {
    for (int r = 0; r < rows; ++r) {
        float sq_sum = 0.f;
        for (int c = 0; c < cols; ++c) {
            float val = x[r * cols + c];
            sq_sum += val * val;
        }
        float rms = 1.f / sqrtf(sq_sum / cols + eps);
        for (int c = 0; c < cols; ++c)
            y[r * cols + c] = x[r * cols + c] * rms * weight[c];
    }
}

int main() {
    constexpr int kRepeat = 10;
    constexpr int kTestCases = 5;
    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/rmsnorm_v1_results.csv");
    ofs << "id,rows,cols,gpu_ms,bandwidth_gb_s,max_abs_diff,check\n";

    for (int i = 0; i < kTestCases; ++i) {
        auto cfg = rmsnorm::RandomTestConfig(2026 + i);
        int rows = cfg.rows, cols = cfg.cols, n = rows * cols;
        // 生成随机测试数据
        std::vector<float> x = rmsnorm::RandomMatrix(rows, cols, 2026 + i);      // 输入矩阵 [-100, 100]
        std::vector<float> w = rmsnorm::RandomWeight(cols, 2026 + i + 100);      // weight 向量 [0.5, 1.5]
        std::vector<float> cpu(n), gpu(n);                                        // CPU/GPU 结果缓冲区
        RMSNormCPU(x.data(), cpu.data(), w.data(), rows, cols, EPS);

        float *dx, *dy, *dw;
        CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dw, cols * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dw, w.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

        const std::size_t smem_size = BR * (cols + 1) * sizeof(float);
        dim3 block(BC, BR);
        dim3 grid((rows + BR - 1) / BR);

        RMSNormV1Kernel<<<grid, block, smem_size>>>(dx, dy, dw, rows, cols, EPS);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        CHECK_CUDA(cudaEventRecord(s));
        for (int rep = 0; rep < kRepeat; ++rep) {
            RMSNormV1Kernel<<<grid, block, smem_size>>>(dx, dy, dw, rows, cols, EPS);
        }
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));
        float gpu_ms_total = 0.f;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_total, s, e));
        const float gpu_ms = gpu_ms_total / static_cast<float>(kRepeat);

        CHECK_CUDA(cudaMemcpy(gpu.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = common::CheckEqual(cpu, gpu, 1e-4f);

        const double bytes = static_cast<double>(n) * sizeof(float) * 3.0;
        const double bw = bytes / (static_cast<double>(gpu_ms) * 1e6);

        std::cout << rows << "x" << cols
                  << " | " << std::fixed << std::setprecision(4) << gpu_ms << " ms"
                  << " | " << std::setprecision(1) << bw << " GB/s"
                  << " | " << (ok ? "PASS" : "FAIL") << "\n";

        ofs << i << "," << rows << "," << cols << "," << gpu_ms << "," << bw << ","
            << common::MaxAbsDiff(cpu, gpu) << "," << (ok ? "PASS" : "FAIL") << "\n";

        CHECK_CUDA(cudaEventDestroy(s));
        CHECK_CUDA(cudaEventDestroy(e));
        CHECK_CUDA(cudaFree(dx));
        CHECK_CUDA(cudaFree(dy));
        CHECK_CUDA(cudaFree(dw));
    }
    return 0;
}
