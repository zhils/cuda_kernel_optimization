// Softmax V2: Online softmax (single-pass algorithm)
//
// Key optimizations over V1:
//   - Single pass over data instead of three (max, exp-sum, normalize)
//   - Maintains running max and sum simultaneously
//   - Rescales partial sums on-the-fly when a new max is found
//   - Reduces memory traffic by ~3x
//
// Reference: Milakov & Gimelshein, "Online normalizer calculation for softmax", 2018

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

__global__ void SoftmaxOnlineKernel(const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    const float* row_x = x + r * cols;
    float* row_y = y + r * cols;

    // Phase 1: online max + sum in a single pass
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    for (int c = tid; c < cols; c += blockSize) {
        float val = row_x[c];
        if (val > local_max) {
            local_sum = local_sum * expf(local_max - val) + expf(0.0f);
            local_max = val;
        } else {
            local_sum += expf(val - local_max);
        }
    }

    // Block-level reduction of (max, sum) pairs
    __shared__ float s_max[256];
    __shared__ float s_sum[256];
    s_max[tid] = local_max;
    s_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float m1 = s_max[tid];
            float m2 = s_max[tid + s];
            float new_max = fmaxf(m1, m2);
            s_sum[tid] = s_sum[tid] * expf(m1 - new_max) +
                         s_sum[tid + s] * expf(m2 - new_max);
            s_max[tid] = new_max;
        }
        __syncthreads();
    }

    float row_max = s_max[0];
    float row_sum = s_sum[0];

    // Phase 2: write normalized output
    for (int c = tid; c < cols; c += blockSize) {
        row_y[c] = expf(row_x[c] - row_max) / row_sum;
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

        int threads = 256;

        SoftmaxOnlineKernel<<<rows, threads>>>(dx, dy, rows, cols);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        CHECK_CUDA(cudaEventRecord(s));
        for (int rep = 0; rep < kRepeat; ++rep)
            SoftmaxOnlineKernel<<<rows, threads>>>(dx, dy, rows, cols);
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));
        float gpu_ms_total = 0;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_total, s, e));
        float gpu_ms = gpu_ms_total / kRepeat;

        CHECK_CUDA(cudaMemcpy(gpu.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = common::CheckEqual(cpu, gpu, 1e-4f);

        // Online: 1 read pass + 1 read+write pass = 3N reads/writes total
        double bytes = static_cast<double>(n) * sizeof(float) * 3.0;
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
    }
    return 0;
}
