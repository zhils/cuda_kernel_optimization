// Softmax V2: Online softmax (single-pass algorithm)
//
// Key optimizations over V1:
//   - Single pass over data instead of three (max, exp-sum, normalize)
//   - Maintains running max and sum simultaneously
//   - Rescales partial sums on-the-fly when a new max is found
//   - Reduces memory traffic by ~3x
//   - Uses __ldg (read-only cache) for global memory reads
//   - Uses float4 vectorized loads when memory is 16-byte aligned
//
// Reference: Milakov & Gimelshein, "Online normalizer calculation for softmax", 2018

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

namespace {

constexpr int kThreads = 256;

__device__ inline float4 LoadX4(const float* __restrict__ ptr) {
    return __ldg(reinterpret_cast<const float4*>(ptr));
}

__device__ inline void OnlineMaxSum4(const float4& v, float& local_max, float& local_sum) {
    float vals[4] = {v.x, v.y, v.z, v.w};
    for (int i = 0; i < 4; ++i) {
        float val = vals[i];
        if (val > local_max) {
            local_sum = local_sum * expf(local_max - val) + expf(0.0f);
            local_max = val;
        } else {
            local_sum += expf(val - local_max);
        }
    }
}

__global__ void SoftmaxOnlineKernel(const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    const float* row_x = x + static_cast<std::size_t>(r) * cols;
    float* row_y = y + static_cast<std::size_t>(r) * cols;

    const bool align4 = ((reinterpret_cast<std::uintptr_t>(row_x) % 16u) == 0u) && ((cols % 4) == 0);

    float local_max = -INFINITY;
    float local_sum = 0.0f;

    if (align4) {
        int c = tid * 4;
        for (; c + 3 < cols; c += blockSize * 4) {
            const float4 v = LoadX4(row_x + c);
            OnlineMaxSum4(v, local_max, local_sum);
        }
        for (int c = c; c < cols; ++c) {
            float val = __ldg(row_x + c);
            if (val > local_max) {
                local_sum = local_sum * expf(local_max - val) + 1.0f;
                local_max = val;
            } else {
                local_sum += expf(val - local_max);
            }
        }
    } else {
        for (int c = tid; c < cols; c += blockSize) {
            float val = __ldg(row_x + c);
            if (val > local_max) {
                local_sum = local_sum * expf(local_max - val) + 1.0f;
                local_max = val;
            } else {
                local_sum += expf(val - local_max);
            }
        }
    }

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

    if (align4) {
        const float inv = 1.0f / row_sum;
        int c = tid * 4;
        for (; c + 3 < cols; c += blockSize * 4) {
            const float4 v = LoadX4(row_x + c);
            const float4 out = make_float4(
                expf(v.x - row_max) * inv,
                expf(v.y - row_max) * inv,
                expf(v.z - row_max) * inv,
                expf(v.w - row_max) * inv
            );
            *reinterpret_cast<float4*>(row_y + c) = out;
        }
        for (int c2 = c; c2 < cols; ++c2) {
            row_y[c2] = expf(__ldg(row_x + c2) - row_max) * inv;
        }
    } else {
        const float inv = 1.0f / row_sum;
        for (int c = tid; c < cols; c += blockSize) {
            row_y[c] = expf(__ldg(row_x + c) - row_max) * inv;
        }
    }
}

inline bool IsAligned4(int cols) {
    return (cols % 4) == 0;
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

        double bytes = static_cast<double>(n) * sizeof(float) * 3.0;
        double bw = bytes / (gpu_ms * 1e6);

        std::cout << rows << "x" << cols << (IsAligned4(cols) ? " [align4]" : " [scalar]")
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
