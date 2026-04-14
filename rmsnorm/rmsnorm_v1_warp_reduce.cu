// RMSNorm V1: Warp-level shuffle reduction
//
// Key optimizations over V0:
//   - One block per row with parallel reduction for sq_sum
//   - __shfl_sync warp-level reduction (no shared memory for reduction)
//   - All threads participate in reading, one warp computes final rms

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

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_sync(0xffffffff, val, offset);
    return val;
}

__global__ void RMSNormWarpReduceKernel(const float* __restrict__ x,
                                         float* __restrict__ y,
                                         const float* __restrict__ weight,
                                         int rows, int cols, float eps) {
    int tid = threadIdx.x;
    int r = blockIdx.x;
    if (r >= rows) return;

    int offset = r * cols;
    float sq_sum = 0.f;

    for (int c = tid; c < cols; c += blockDim.x) {
        float val = x[offset + c];
        sq_sum += val * val;
    }

    sq_sum = warpReduceSum(sq_sum);

    __shared__ float rms_val;
    if (tid == 0) {
        rms_val = rsqrtf(sq_sum / cols + eps);
    }
    __syncthreads();

    float rms = rms_val;
    for (int c = tid; c < cols; c += blockDim.x) {
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
    std::ofstream ofs("data/results/rmsnorm_v1_results.csv");
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
        RMSNormWarpReduceKernel<<<rows, threads>>>(dx, dy, dw, rows, cols, kEps);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        CHECK_CUDA(cudaEventRecord(s));
        for (int rep = 0; rep < kRepeat; ++rep)
            RMSNormWarpReduceKernel<<<rows, threads>>>(dx, dy, dw, rows, cols, kEps);
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
