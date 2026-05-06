// Softmax V2: 在 V1 基础上使用 Warp 归约
//
// 核心设计:
// - 1维 block: 128 线程 = 4 个 warp，每个 warp 处理一行数据
// - 共享内存缓存: exp 结果存入共享内存，避免全局内存中间读写
// - Warp 归约: 使用 __shfl_down_sync 进行 warp shuffle 归约
// - float4 向量化: 使用 float4 加载和写回，提升内存带宽

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

__global__ void SoftmaxV2Kernel(
    const float* __restrict__ x, 
    float* __restrict__ y,
    int rows, int cols
) {
    
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

    float local_max = -INFINITY;
    for (int c = lane; c < cols; c += WARP_SIZE) {
        local_max = fmaxf(local_max, s_exp[c]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane == 0) s_max[warp_id] = local_max;
    __syncthreads();
    const float row_max = s_max[warp_id];

    float local_sum = 0.f;
    for (int c = lane; c < cols; c += WARP_SIZE) {
        const float e = __expf(s_exp[c] - row_max);
        s_exp[c] = e;
        local_sum += e;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane == 0) s_sum[warp_id] = local_sum;
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
    std::ofstream ofs("data/results/softmax_v2_results.csv");
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
        CHECK_CUDA(cudaFuncSetAttribute(SoftmaxV2Kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));

        dim3 block(BLOCK_SIZE);
        dim3 grid((rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
        SoftmaxV2Kernel<<<grid, block, smem_size>>>(dx, dy, rows, cols);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        CHECK_CUDA(cudaEventRecord(s));
        for (int rep = 0; rep < kRepeat; ++rep) {
            SoftmaxV2Kernel<<<grid, block, smem_size>>>(dx, dy, rows, cols);
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
