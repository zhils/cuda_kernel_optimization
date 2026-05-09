// Softmax V3
//
// 这一版的重点是两件事：
// 1) Online Softmax：单遍同时维护 row_max 和 row_sum
// 2) 向量化 + 共享内存：先把一行搬到 smem，再做归约和写回
//
// 对比 v2：
// - v2 是两遍（max 一遍 + sum 一遍）
// - v3 把两遍合成一遍，减少同步和访存

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

__global__ void SoftmaxV3Kernel(
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
    float* s_data = smem + warp_id * cols;
    __shared__ float s_max[WARPS_PER_BLOCK];
    __shared__ float s_sum[WARPS_PER_BLOCK];

    const int cols4 = cols / 4;

    #pragma unroll 4
    for (int c = lane; c < cols4; c += WARP_SIZE) {
        const float4 v = __ldg(reinterpret_cast<const float4*>(row_x + c * 4));
        s_data[c * 4 + 0] = v.x;
        s_data[c * 4 + 1] = v.y;
        s_data[c * 4 + 2] = v.z;
        s_data[c * 4 + 3] = v.w;
    }
    for (int c = cols4 * 4 + lane; c < cols; c += WARP_SIZE) {
        s_data[c] = __ldg(row_x + c);
    }
    __syncthreads();

    float local_max = -INFINITY;
    float local_sum = 0.f;

    #pragma unroll 8
    for (int c = lane; c < cols; c += WARP_SIZE) {
        const float val = s_data[c];
        if (val > local_max) {
            local_sum *= __expf(local_max - val);
            local_max = val;
        }
        local_sum += __expf(val - local_max);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m2 = __shfl_down_sync(0xffffffff, local_max, offset);
        float s2 = __shfl_down_sync(0xffffffff, local_sum, offset);
        float new_max = fmaxf(local_max, m2);
        local_sum = local_sum * __expf(local_max - new_max) + s2 * __expf(m2 - new_max);
        local_max = new_max;
    }
    if (lane == 0) {
        s_max[warp_id] = local_max;
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();

    const float row_max = s_max[warp_id];
    const float row_sum = s_sum[warp_id];
    const float inv = 1.f / row_sum;

    #pragma unroll 4
    for (int c = lane; c < cols4; c += WARP_SIZE) {
        const float4 d = *reinterpret_cast<const float4*>(s_data + c * 4);
        float4 out;
        out.x = __expf(d.x - row_max) * inv;
        out.y = __expf(d.y - row_max) * inv;
        out.z = __expf(d.z - row_max) * inv;
        out.w = __expf(d.w - row_max) * inv;
        *reinterpret_cast<float4*>(row_y + c * 4) = out;
    }
    for (int c = cols4 * 4 + lane; c < cols; c += WARP_SIZE) {
        row_y[c] = __expf(s_data[c] - row_max) * inv;
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
    std::ofstream ofs("data/results/softmax_v3_results.csv");
    ofs << "id,rows,cols,gpu_ms,bandwidth_gb_s,max_abs_diff,check\n";

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    size_t max_smem = prop.sharedMemPerBlock;

    cudaFuncAttributes attr;
    CHECK_CUDA(cudaFuncGetAttributes(&attr, SoftmaxV3Kernel));
    size_t max_dyn_smem = max_smem;
    if (prop.major >= 12) {
        max_dyn_smem = 96 * 1024;
        cudaError_t err = cudaFuncSetAttribute(SoftmaxV3Kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                max_dyn_smem);
        if (err != cudaSuccess) {
            max_dyn_smem = max_smem;
        }
    }

    for (size_t i = 0; i < cases.size(); ++i) {
        int rows = cases[i].rows, cols = cases[i].cols, n = rows * cols;
        size_t smem_size = WARPS_PER_BLOCK * cols * sizeof(float);
        if (smem_size > max_dyn_smem) {
            std::cout << rows << "x" << cols << " | Skipped (smem " << smem_size/1024 << "KB > max " << max_dyn_smem/1024 << "KB)\n";
            continue;
        }

        std::vector<float> x(n), cpu(n), gpu(n);
        common::InitMatrix(x, rows, cols);
        SoftmaxCPU(x.data(), cpu.data(), rows, cols);

        float *dx, *dy;
        CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(BLOCK_SIZE);
        dim3 grid((rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
        SoftmaxV3Kernel<<<grid, block, smem_size>>>(dx, dy, rows, cols);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        CHECK_CUDA(cudaEventRecord(s));
        for (int rep = 0; rep < kRepeat; ++rep) {
            SoftmaxV3Kernel<<<grid, block, smem_size>>>(dx, dy, rows, cols);
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
