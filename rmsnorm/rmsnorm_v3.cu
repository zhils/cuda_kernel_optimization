// RMSNorm V3: Weight in shared memory, x from global memory
// Fixed block size = 64

#include <cuda_runtime.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#endif

#include "../common/include/common/benchmark.h"
#include "../common/include/common/cuda_utils.h"
#include "test_utils.h"

#define EPS 1e-5f
#define WARP_SIZE 32
#define BLOCK_SIZE 128
#define WARPS_PER_BLOCK 4

__global__ void RMSNormV3Kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    int rows, int cols, float eps
) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    extern __shared__ float s_weight[];
    for (int c = threadIdx.x; c < cols; c += BLOCK_SIZE) {
        s_weight[c] = weight[c];
    }
    __syncthreads();

    if (row >= rows) return;

    const float* row_x = x + row * cols;
    float* row_y = y + row * cols;

    const int cols4 = cols / 4;
    float local_sum = 0.f;

    for (int c = lane; c < cols4; c += WARP_SIZE) {
        const float4 v = __ldg(reinterpret_cast<const float4*>(row_x + c * 4));
        local_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (int c = cols4 * 4 + lane; c < cols; c += WARP_SIZE) {
        const float val = __ldg(row_x + c);
        local_sum += val * val;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    const float sq_sum = __shfl_sync(0xffffffff, local_sum, 0);
    const float rms = rsqrtf(sq_sum / static_cast<float>(cols) + eps);

    for (int c = lane; c < cols4; c += WARP_SIZE) {
        const float4 vx = __ldg(reinterpret_cast<const float4*>(row_x + c * 4));
        const float4 vw = *reinterpret_cast<const float4*>(s_weight + c * 4);
        float4 vy;
        vy.x = vx.x * rms * vw.x;
        vy.y = vx.y * rms * vw.y;
        vy.z = vx.z * rms * vw.z;
        vy.w = vx.w * rms * vw.w;
        *reinterpret_cast<float4*>(row_y + c * 4) = vy;
    }
    for (int c = cols4 * 4 + lane; c < cols; c += WARP_SIZE) {
        row_y[c] = __ldg(row_x + c) * rms * s_weight[c];
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
#ifdef _WIN32
    _mkdir("data");
    _mkdir("data\\results");
#else
    mkdir("data", 0755);
    mkdir("data/results", 0755);
#endif
    std::ofstream ofs("data/results/rmsnorm_v3_results.csv");
    ofs << "id,rows,cols,gpu_ms,bandwidth_gb_s,max_abs_diff,check\n";

    std::vector<std::pair<int, int>> test_sizes = {
        {128, 128}, {256, 256}, {512, 512}, {1024, 1024}, {4096, 4096}
    };

    for (int i = 0; i < kTestCases; ++i) {
        int rows = test_sizes[i].first;
        int cols = test_sizes[i].second;
        int n = rows * cols;
        std::vector<float> x = rmsnorm::RandomMatrix(rows, cols, 2026 + i);
        std::vector<float> w = rmsnorm::RandomWeight(cols, 2026 + i + 100);
        std::vector<float> cpu(n), gpu(n);
        RMSNormCPU(x.data(), cpu.data(), w.data(), rows, cols, EPS);

        float *dx, *dy, *dw;
        CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dw, cols * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dw, w.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

        const size_t smem_size = cols * sizeof(float);
        dim3 block(BLOCK_SIZE);
        dim3 grid((rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

        cudaFuncSetAttribute(RMSNormV3Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        RMSNormV3Kernel<<<grid, block, smem_size>>>(dx, dy, dw, rows, cols, EPS);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        CHECK_CUDA(cudaEventRecord(s));
        for (int rep = 0; rep < kRepeat; ++rep) {
            RMSNormV3Kernel<<<grid, block, smem_size>>>(dx, dy, dw, rows, cols, EPS);
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
                  << " | BlockSize=" << BLOCK_SIZE
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
