// RMSNorm V2: Vectorized memory access (float4) + warp reduction
//
// Key optimizations over V1:
//   - float4 vectorized loads for sq_sum computation (4x fewer transactions)
//   - float4 vectorized stores for output writing
//   - All threads participate in both reduction and output writing
//   - Shared memory for cross-warp reduction

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

static constexpr float kEps = 1e-5f;
static constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_sync(0xffffffff, val, offset);
    return val;
}

__global__ void RMSNormVectorizedKernel(const float* __restrict__ x,
                                         float* __restrict__ y,
                                         const float* __restrict__ weight,
                                         int rows, int cols, float eps) {
    int tid = threadIdx.x;
    int r = blockIdx.x;
    if (r >= rows) return;

    int offset = r * cols;
    int n4 = cols / 4;

    const float4* x4 = reinterpret_cast<const float4*>(x + offset);
    float4* y4 = reinterpret_cast<float4*>(y + offset);

    float sq_sum = 0.f;
    for (int c = tid; c < n4; c += blockDim.x) {
        float4 val4 = x4[c];
        sq_sum += val4.x * val4.x + val4.y * val4.y +
                  val4.z * val4.z + val4.w * val4.w;
    }

    sq_sum = warpReduceSum(sq_sum);

    // Cross-warp reduction via shared memory
    __shared__ float warp_sums[8];
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    if (lane == 0) warp_sums[warp_id] = sq_sum;
    __syncthreads();

    if (warp_id == 0) {
        sq_sum = (lane < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE)
                     ? warp_sums[lane]
                     : 0.f;
        sq_sum = warpReduceSum(sq_sum);
    }

    __shared__ float rms_val;
    if (tid == 0) rms_val = rsqrtf(sq_sum / cols + eps);
    __syncthreads();

    float rms = rms_val;

    for (int c = tid; c < n4; c += blockDim.x) {
        float4 val4 = x4[c];
        float4 res4;
        res4.x = val4.x * rms * weight[c * 4];
        res4.y = val4.y * rms * weight[c * 4 + 1];
        res4.z = val4.z * rms * weight[c * 4 + 2];
        res4.w = val4.w * rms * weight[c * 4 + 3];
        y4[c] = res4;
    }

    for (int c = n4 * 4 + tid; c < cols; c += blockDim.x) {
        y[offset + c] = x[offset + c] * rms * weight[c];
    }
}

static void RMSNormCPU(const float* x, float* y, const float* weight,
                        int rows, int cols, float eps) {
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
    auto cases = common::LoadOrCreateTestCasesCsv("data/rmsnorm/test_cases.csv");
    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/rmsnorm_v2_results.csv");
    ofs << "id,rows,cols,gpu_ms,bandwidth_gb_s,max_abs_diff,check\n";

    for (size_t i = 0; i < cases.size(); ++i) {
        int rows = cases[i].rows, cols = cases[i].cols, n = rows * cols;
        std::vector<float> x(n), w(cols), cpu(n), gpu(n);
        common::InitMatrix(x, rows, cols);
        common::InitMatrix(w, 1, cols);
        RMSNormCPU(x.data(), cpu.data(), w.data(), rows, cols, kEps);

        float *dx, *dy, *dw;
        CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dw, cols * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dw, w.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

        int threads = 256;
        RMSNormVectorizedKernel<<<rows, threads>>>(dx, dy, dw, rows, cols, kEps);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        CHECK_CUDA(cudaEventRecord(s));
        for (int rep = 0; rep < kRepeat; ++rep)
            RMSNormVectorizedKernel<<<rows, threads>>>(dx, dy, dw, rows, cols, kEps);
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));
        float gpu_ms_total = 0;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_total, s, e));
        float gpu_ms = gpu_ms_total / kRepeat;

        CHECK_CUDA(cudaMemcpy(gpu.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = common::CheckEqual(cpu, gpu, 1e-4f);

        double bytes = static_cast<double>(n) * sizeof(float) * 2.0 + cols * sizeof(float);
        double bw = bytes / (gpu_ms * 1e6);

        std::cout << rows << "x" << cols
                  << " | " << std::fixed << std::setprecision(4) << gpu_ms << " ms"
                  << " | " << std::setprecision(1) << bw << " GB/s"
                  << " | " << (ok ? "PASS" : "FAIL") << "\n";

        ofs << i << "," << rows << "," << cols << ","
            << gpu_ms << "," << bw << ","
            << common::MaxAbsDiff(cpu, gpu) << "," << (ok ? "PASS" : "FAIL") << "\n";

        CHECK_CUDA(cudaEventDestroy(s));
        CHECK_CUDA(cudaEventDestroy(e));
        CHECK_CUDA(cudaFree(dx));
        CHECK_CUDA(cudaFree(dy));
        CHECK_CUDA(cudaFree(dw));
    }
    return 0;
}
