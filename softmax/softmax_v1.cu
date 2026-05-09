// Softmax V1
//
// 核心设计：
// - 1D block（128 线程 = 4 warp），每个 warp 对应一行
// - 数据先放共享内存，减少中间访存
// - lane 0 串行完成该行 max/sum
// - float4 向量化加载与写回

#include <cuda_runtime.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)

__global__ void SoftmaxV1Kernel(const float* __restrict__ x, float* __restrict__ y,
                                int rows, int cols) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= rows) return;

    const float* row_x = x + row * cols;
    float* row_y = y + row * cols;

    extern __shared__ float smem[];
    float* s_exp = smem + warp_id * cols;
    __shared__ float s_max[WARPS_PER_BLOCK];
    __shared__ float s_sum[WARPS_PER_BLOCK];

    const int cols4 = cols / 4;

    for (int c = lane; c < cols4; c += WARP_SIZE) {
        const float4 v = __ldg(reinterpret_cast<const float4*>(row_x + c * 4));
        s_exp[c * 4 + 0] = v.x;
        s_exp[c * 4 + 1] = v.y;
        s_exp[c * 4 + 2] = v.z;
        s_exp[c * 4 + 3] = v.w;
    }
    for (int c = cols4 * 4 + lane; c < cols; c += WARP_SIZE) {
        s_exp[c] = __ldg(row_x + c);
    }
    __syncthreads();

    if (lane == 0) {
        float row_max = s_exp[0];
        for (int c = 1; c < cols; ++c) {
            row_max = fmaxf(row_max, s_exp[c]);
        }
        s_max[warp_id] = row_max;

        float row_sum = 0.f;
        for (int c = 0; c < cols; ++c) {
            s_exp[c] = __expf(s_exp[c] - row_max);
            row_sum += s_exp[c];
        }
        s_sum[warp_id] = row_sum;
    }
    __syncthreads();

    const float inv = 1.f / s_sum[warp_id];
    for (int c = lane; c < cols4; c += WARP_SIZE) {
        const float4 e = *reinterpret_cast<const float4*>(s_exp + c * 4);
        float4 out;
        out.x = e.x * inv;
        out.y = e.y * inv;
        out.z = e.z * inv;
        out.w = e.w * inv;
        *reinterpret_cast<float4*>(row_y + c * 4) = out;
    }
    for (int c = cols4 * 4 + lane; c < cols; c += WARP_SIZE) {
        row_y[c] = s_exp[c] * inv;
    }
}

#ifndef PYTORCH_EXTENSION

static void SoftmaxCPU(const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float maxv = x[r * cols];
        for (int c = 1; c < cols; ++c) maxv = std::max(maxv, x[r * cols + c]);
        float sum = 0.f;
        for (int c = 0; c < cols; ++c) {
            float v = std::exp(x[r * cols + c] - maxv);
            y[r * cols + c] = v;
            sum += v;
        }
        for (int c = 0; c < cols; ++c) y[r * cols + c] /= sum;
    }
}

int main() {
    constexpr int kRepeat = 10;
    auto cases = common::LoadOrCreateTestCasesCsv("data/softmax/test_cases.csv");
    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/softmax_v1_results.csv");
    ofs << "id,rows,cols,gpu_ms,bandwidth_gb_s,max_abs_diff,check\n";

    for (size_t i = 0; i < cases.size(); ++i) {
        int rows = cases[i].rows, cols = cases[i].cols, n = rows * cols;
        std::vector<float> x(n), cpu(n), gpu(n);
        common::InitMatrix(x, rows, cols);
        SoftmaxCPU(x.data(), cpu.data(), rows, cols);

        float *dx, *dy;
        CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        const size_t smem_size = WARPS_PER_BLOCK * cols * sizeof(float);
        CHECK_CUDA(cudaFuncSetAttribute(SoftmaxV1Kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));

        dim3 block(BLOCK_SIZE);
        dim3 grid((rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
        SoftmaxV1Kernel<<<grid, block, smem_size>>>(dx, dy, rows, cols);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        CHECK_CUDA(cudaEventRecord(s));
        for (int rep = 0; rep < kRepeat; ++rep) {
            SoftmaxV1Kernel<<<grid, block, smem_size>>>(dx, dy, rows, cols);
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
    }
    return 0;
}

#endif  // PYTORCH_EXTENSION
