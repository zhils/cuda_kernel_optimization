// RMSNorm V2: 在 V1 基础上, 使用 Warp 归约计算 sq_sum
//
// 核心设计:
// — 1维 block: 128 线程 = 4 个 warp，每个 warp 处理一行数据
// — 共享内存 staging: 数据先加载到共享内存，后续计算从共享内存读取
// — warp 归约: 使用 warp shuffle 并行计算 sq_sum
// — float4 向量化: 使用 float4 加载和写回，提升内存带宽

#include <cuda_runtime.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../common/include/common/benchmark.h"
#include "../common/include/common/cuda_utils.h"
#include "test_utils.h"

#define EPS 1e-5f
#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4

// ---------- 1D Block Kernel: 每个 warp 处理一行 ----------
__global__ void RMSNormV2Kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    int rows, int cols, float eps
) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= rows) return;

    const float* row_x = x + row * cols;
    float* row_y = y + row * cols;

    // 静态共享内存: 存储每个 warp 的 sq_sum (4 个 float)
    __shared__ float s_sq_sum[WARPS_PER_BLOCK];
    // 动态共享内存: 存储 4 行数据，每行 cols 个 float
    extern __shared__ float s_data[];
    float* s_row = s_data + warp_id * cols;  // 每个 warp 指向自己的行

    // Step 1: 每个 warp 协作加载一行数据到共享内存 (float4 向量化)
    const int cols4 = cols / 4;
    for (int c = lane; c < cols4; c += WARP_SIZE) {
        const float4 v = *reinterpret_cast<const float4*>(row_x + c * 4);
        s_row[c * 4 + 0] = v.x;
        s_row[c * 4 + 1] = v.y;
        s_row[c * 4 + 2] = v.z;
        s_row[c * 4 + 3] = v.w;
    }
    // 处理尾巴 (cols 不是 4 的倍数)
    for (int c = cols4 * 4 + lane; c < cols; c += WARP_SIZE) {
        s_row[c] = row_x[c];
    }
    __syncthreads();

    // Step 2: Warp 归约计算平方和 (从共享内存读取)
    // 每个线程计算部分和
    float local_sum = 0.f;
    for (int c = lane; c < cols; c += WARP_SIZE) {
        const float val = s_row[c];
        local_sum += val * val;
    }
    // Warp shuffle 归约
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    // lane 0 写入结果
    if (lane == 0) {
        s_sq_sum[warp_id] = local_sum;
    }
    __syncthreads();

    // Step 3: 计算 RMS 并写回结果 (从共享内存读取，float4 向量化写回)
    const float rms = rsqrtf(s_sq_sum[warp_id] / static_cast<float>(cols) + eps);
    for (int c = lane; c < cols4; c += WARP_SIZE) {
        const float4 vx = *reinterpret_cast<const float4*>(s_row + c * 4);
        const float4 vw = *reinterpret_cast<const float4*>(weight + c * 4);
        float4 vy;
        vy.x = vx.x * rms * vw.x;
        vy.y = vx.y * rms * vw.y;
        vy.z = vx.z * rms * vw.z;
        vy.w = vx.w * rms * vw.w;
        *reinterpret_cast<float4*>(row_y + c * 4) = vy;
    }
    // 处理尾巴 (cols 不是 4 的倍数)
    for (int c = cols4 * 4 + lane; c < cols; c += WARP_SIZE) {
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
    std::ofstream ofs("data/results/rmsnorm_v2_results.csv");
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

        const size_t smem_size = WARPS_PER_BLOCK * cols * sizeof(float);
        dim3 block(BLOCK_SIZE);
        dim3 grid((rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

        cudaFuncSetAttribute(RMSNormV2Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        RMSNormV2Kernel<<<grid, block, smem_size>>>(dx, dy, dw, rows, cols, EPS);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        CHECK_CUDA(cudaEventRecord(s));
        for (int rep = 0; rep < kRepeat; ++rep) {
            RMSNormV2Kernel<<<grid, block, smem_size>>>(dx, dy, dw, rows, cols, EPS);
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
