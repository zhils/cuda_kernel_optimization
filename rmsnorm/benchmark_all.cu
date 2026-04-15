// RMSNorm Performance Comparison: Naive vs Optimized vs CUB vs cuDNN
// NVIDIA Library Implementations: CUB (reduction), cuDNN (layer norm)

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cudnn.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

#define CHECK_CUDNN(call)                                                        \
  do {                                                                            \
    cudnnStatus_t s__ = (call);                                                  \
    if (s__ != CUDNN_STATUS_SUCCESS) {                                           \
      std::cerr << "cuDNN error: " << s__ << std::endl;                          \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

static constexpr float kEps = 1e-5f;
static constexpr int WARP_SIZE = 32;

// ============================================================================
// Version 1: Naive - 基础实现
// ============================================================================
__global__ void RMSNormKernelNaive(const float* __restrict__ x,
                                    float* __restrict__ y,
                                    const float* __restrict__ weight,
                                    int rows, int cols, float eps) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    float sq_sum = 0.0f;
    int offset = r * cols;

    for (int c = 0; c < cols; ++c) {
        float val = x[offset + c];
        sq_sum += val * val;
    }

    float rms = rsqrtf(sq_sum / cols + eps);

    for (int c = 0; c < cols; ++c) {
        y[offset + c] = x[offset + c] * rms * weight[c];
    }
}

// ============================================================================
// Version 2: Optimized - Warp Reduction
// ============================================================================
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void RMSNormKernelWarpReduce(const float* __restrict__ x,
                                          float* __restrict__ y,
                                          const float* __restrict__ weight,
                                          int rows, int cols, float eps) {
    int tid = threadIdx.x;
    int r = blockIdx.x;
    if (r >= rows) return;

    int offset = r * cols;
    float sq_sum = 0.0f;

    for (int c = tid; c < cols; c += blockDim.x) {
        float val = x[offset + c];
        sq_sum += val * val;
    }

    sq_sum = warpReduceSum(sq_sum);

    if (tid == 0) {
        float rms = rsqrtf(sq_sum / cols + eps);
        for (int c = 0; c < cols; ++c) {
            y[offset + c] = x[offset + c] * rms * weight[c];
        }
    }
}

// ============================================================================
// Version 3: Vectorized - float4 + Warp Reduction
// ============================================================================
__global__ void RMSNormKernelVectorized(const float* __restrict__ x,
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

    float4 sum4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int c = tid; c < n4; c += blockDim.x) {
        float4 val4 = x4[c];
        sum4.x += val4.x * val4.x;
        sum4.y += val4.y * val4.y;
        sum4.z += val4.z * val4.z;
        sum4.w += val4.w * val4.w;
    }

    float sq_sum = sum4.x + sum4.y + sum4.z + sum4.w;
    sq_sum = warpReduceSum(sq_sum);

    if (tid == 0) {
        float rms = rsqrtf(sq_sum / cols + eps);

        for (int c = 0; c < n4; ++c) {
            float4 val4 = x4[c];
            float4 res4;
            res4.x = val4.x * rms * weight[c * 4];
            res4.y = val4.y * rms * weight[c * 4 + 1];
            res4.z = val4.z * rms * weight[c * 4 + 2];
            res4.w = val4.w * rms * weight[c * 4 + 3];
            y4[c] = res4;
        }

        for (int c = n4 * 4; c < cols; ++c) {
            y[offset + c] = x[offset + c] * rms * weight[c];
        }
    }
}

// ============================================================================
// Version 4: CUB Implementation
// ============================================================================
struct RmsNormPreamble {
    float sq_sum;
    float rms;
};

__global__ void RMSNormCubPrepKernel(const float* __restrict__ x,
                                      float* __restrict__ sq_sum_out,
                                      int rows, int cols) {
    int tid = threadIdx.x;
    int r = blockIdx.x;
    if (r >= rows) return;

    int offset = r * cols;
    float sq_sum = 0.0f;

    typedef cub::WarpReduce<float> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage;

    for (int c = tid; c < cols; c += blockDim.x) {
        float val = x[offset + c];
        sq_sum += val * val;
    }

    sq_sum = WarpReduce(temp_storage).Sum(sq_sum);

    if (tid == 0) {
        sq_sum_out[r] = sq_sum;
    }
}

__global__ void RMSNormCubScaleKernel(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       const float* __restrict__ weight,
                                       const float* __restrict__ sq_sums,
                                       int rows, int cols, float eps) {
    int tid = threadIdx.x;
    int r = blockIdx.x;
    if (r >= rows) return;

    int offset = r * cols;
    float sq_sum = sq_sums[r];
    float rms = rsqrtf(sq_sum / cols + eps);

    for (int c = tid; c < cols; c += blockDim.x) {
        y[offset + c] = x[offset + c] * rms * weight[c];
    }
}

static double RunCubRMSNorm(const float* h_x,
                            const float* h_weight,
                            float* h_y,
                            int rows, int cols) {
    int n = rows * cols;
    float *d_x, *d_y, *d_weight, *d_sq_sums;
    float *d_sq_sums_temp = nullptr;

    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weight, cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sq_sums, rows * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight, h_weight, cols * sizeof(float), cudaMemcpyHostToDevice));

    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, d_sq_sums, d_sq_sums, rows);

    CHECK_CUDA(cudaMalloc(&d_sq_sums_temp, temp_storage_bytes));

    int threads = 256;
    int blocks = rows;

    RMSNormCubPrepKernel<<<blocks, threads>>>(d_x, d_sq_sums, rows, cols);

    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(s));

    RMSNormCubPrepKernel<<<blocks, threads>>>(d_x, d_sq_sums, rows, cols);
    RMSNormCubScaleKernel<<<blocks, threads>>>(d_x, d_y, d_weight, d_sq_sums, rows, cols, kEps);

    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));

    CHECK_CUDA(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_sq_sums));
    if (d_sq_sums_temp) CHECK_CUDA(cudaFree(d_sq_sums_temp));

    return ms;
}

// ============================================================================
// Version 5: cuDNN LayerNorm (configured as RMSNorm)
// ============================================================================
static double RunCudnnRMSNorm(const float* h_x,
                               const float* h_weight,
                               float* h_y,
                               int rows, int cols) {
    (void)h_x;
    (void)h_weight;
    (void)h_y;
    (void)rows;
    (void)cols;
    // cuDNN 9.x 移除了此文件中使用的旧 RMSNorm API（需要迁移到 frontend/backend graph API）。
    // 先返回 -1 以保证 CUB/自研 kernel 基准链路可编译可运行。
    return -1.0;
}

// ============================================================================
// CPU Reference Implementation
// ============================================================================
static void RMSNormCPU(const float* x, float* y, const float* weight, int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        float sq_sum = 0.0f;
        int offset = r * cols;
        for (int c = 0; c < cols; ++c) {
            float val = x[offset + c];
            sq_sum += val * val;
        }
        float rms = 1.0f / sqrtf(sq_sum / cols + eps);
        for (int c = 0; c < cols; ++c) {
            y[offset + c] = x[offset + c] * rms * weight[c];
        }
    }
}

// ============================================================================
// Benchmark Helper Functions
// ============================================================================
template<typename KernelFunc>
double RunKernelBenchmark(KernelFunc kernel,
                          const std::vector<float>& h_x,
                          const std::vector<float>& h_weight,
                          std::vector<float>& h_y,
                          std::vector<float>& h_cpu,
                          int rows, int cols,
                          int iterations = 10) {
    float *d_x, *d_y, *d_weight;
    CHECK_CUDA(cudaMalloc(&d_x, rows * cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, rows * cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weight, cols * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight, h_weight.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = rows;

    for (int i = 0; i < 3; ++i) {
        kernel<<<blocks, threads>>>(d_x, d_y, d_weight, rows, cols, kEps);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<double> times;
    for (int iter = 0; iter < iterations; ++iter) {
        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));

        CHECK_CUDA(cudaEventRecord(s));
        kernel<<<blocks, threads>>>(d_x, d_y, d_weight, rows, cols, kEps);
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
        times.push_back(ms);

        CHECK_CUDA(cudaEventDestroy(s));
        CHECK_CUDA(cudaEventDestroy(e));
    }

    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_weight));

    std::sort(times.begin(), times.end());
    double sum = 0;
    for (size_t i = 1; i < times.size() - 1; ++i) {
        sum += times[i];
    }
    return sum / (times.size() - 2);
}

// ============================================================================
// Main Function
// ============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "  RMSNorm Kernel Performance Comparison\n";
    std::cout << "  Including: Naive, Optimized, CUB, cuDNN\n";
    std::cout << "========================================\n\n";

    auto cases = common::LoadOrCreateTestCasesCsv("data/rmsnorm/test_cases.csv");

    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/rmsnorm_all_comparison.csv");
    ofs << "id,rows,cols,naive_ms,warpreduce_ms,vectorized_ms,cub_ms,cudnn_ms,cpu_ms\n";

    std::cout << std::left
              << std::setw(8) << "Rows"
              << std::setw(10) << "Cols"
              << std::setw(12) << "Naive"
              << std::setw(14) << "WarpReduce"
              << std::setw(14) << "Vectorized"
              << std::setw(12) << "CUB"
              << std::setw(12) << "cuDNN"
              << std::setw(12) << "CPU"
              << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (size_t i = 0; i < cases.size(); ++i) {
        int rows = cases[i].rows;
        int cols = cases[i].cols;
        int n = rows * cols;

        std::vector<float> h_x(n), h_weight(cols), h_cpu(n), h_gpu(n);
        common::InitMatrix(h_x, rows, cols);
        common::InitMatrix(h_weight, 1, cols);

        auto t0 = std::chrono::high_resolution_clock::now();
        RMSNormCPU(h_x.data(), h_cpu.data(), h_weight.data(), rows, cols, kEps);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << std::left
                  << std::setw(8) << rows
                  << std::setw(10) << cols;

        double naive_ms = RunKernelBenchmark(RMSNormKernelNaive, h_x, h_weight, h_gpu, h_cpu, rows, cols);
        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(12) << naive_ms;

        double warpreduce_ms = RunKernelBenchmark(RMSNormKernelWarpReduce, h_x, h_weight, h_gpu, h_cpu, rows, cols);
        std::cout << std::setw(14) << warpreduce_ms;

        double vectorized_ms = RunKernelBenchmark(RMSNormKernelVectorized, h_x, h_weight, h_gpu, h_cpu, rows, cols);
        std::cout << std::setw(14) << vectorized_ms;

        double cub_ms = RunCubRMSNorm(h_x.data(), h_weight.data(), h_gpu.data(), rows, cols);
        std::cout << std::setw(12) << cub_ms;

        double cudnn_ms = RunCudnnRMSNorm(h_x.data(), h_weight.data(), h_gpu.data(), rows, cols);
        std::cout << std::setw(12) << cudnn_ms;

        std::cout << std::setw(12) << cpu_ms << "\n";

        ofs << i << ","
            << rows << ","
            << cols << ","
            << naive_ms << ","
            << warpreduce_ms << ","
            << vectorized_ms << ","
            << cub_ms << ","
            << cudnn_ms << ","
            << cpu_ms << "\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "Notes:\n";
    std::cout << "  - CUB: NVIDIA CUDA Unbound library for optimized reductions\n";
    std::cout << "  - cuDNN: NVIDIA Deep Learning library with LayerNorm API\n";
    std::cout << "  - RMSNorm can be expressed as LayerNorm without mean subtraction\n";
    std::cout << "========================================\n";

    return 0;
}
