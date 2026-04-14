// Softmax V1: Block-level shared memory reduction
//
// Key optimizations over V0:
//   - One block per row (block-level parallelism within each row)
//   - Shared memory for max/sum reduction across threads
//   - Coalesced global memory access

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

__global__ void SoftmaxSharedMemKernel(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       int rows, int cols) {
    extern __shared__ float sdata[];

    int r = blockIdx.x;
    if (r >= rows) return;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float thread_max = -INFINITY;
    for (int c = tid; c < cols; c += blockSize) {
        thread_max = fmaxf(thread_max, x[r * cols + c]);
    }
    sdata[tid] = thread_max;
    __syncthreads();

    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];

    float thread_sum = 0.f;
    for (int c = tid; c < cols; c += blockSize) {
        float v = expf(x[r * cols + c] - row_max);
        y[r * cols + c] = v;
        thread_sum += v;
    }

    sdata[tid] = thread_sum;
    __syncthreads();

    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];

    for (int c = tid; c < cols; c += blockSize) {
        y[r * cols + c] /= row_sum;
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

        int threads = 256;
        size_t smem = threads * sizeof(float);

        SoftmaxSharedMemKernel<<<rows, threads, smem>>>(dx, dy, rows, cols);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        CHECK_CUDA(cudaEventRecord(s));
        for (int rep = 0; rep < kRepeat; ++rep)
            SoftmaxSharedMemKernel<<<rows, threads, smem>>>(dx, dy, rows, cols);
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));
        float gpu_ms_total = 0;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_total, s, e));
        float gpu_ms = gpu_ms_total / kRepeat;

        CHECK_CUDA(cudaMemcpy(gpu.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = common::CheckEqual(cpu, gpu, 1e-4f);

        // 3 passes over data: read(max), read+write(exp), read+write(div)
        double bytes = static_cast<double>(n) * sizeof(float) * 5.0;
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
