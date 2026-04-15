// Softmax Performance Comparison: Naive vs Optimized vs WarpReduce vs CUB vs cuDNN
// NVIDIA Library Implementations: cuDNN, CUB

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

#define CHECK_CUDNN(call)                                                         \
  do {                                                                            \
    cudnnStatus_t s__ = (call);                                                   \
    if (s__ != CUDNN_STATUS_SUCCESS) {                                            \
      std::cerr << "cuDNN error: " << s__ << std::endl;                           \
      std::exit(EXIT_FAILURE);                                                    \
    }                                                                            \
  } while (0)

// ============ Kernel 1: Naive Softmax ============
__global__ void SoftmaxNaiveKernel(const float* __restrict__ x,
                                    float* __restrict__ y,
                                    int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    float maxv = x[r * cols];
    for (int c = 1; c < cols; ++c) {
        maxv = fmaxf(maxv, x[r * cols + c]);
    }

    float sum = 0.f;
    for (int c = 0; c < cols; ++c) {
        float v = expf(x[r * cols + c] - maxv);
        y[r * cols + c] = v;
        sum += v;
    }

    for (int c = 0; c < cols; ++c) {
        y[r * cols + c] /= sum;
    }
}

// ============ Kernel 2: Shared Memory Optimized ============
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

    for (int s = blockSize / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    float row_max = sdata[0];
    for (int i = 1; i < 32; ++i) {
        row_max = fmaxf(row_max, sdata[i]);
    }

    float thread_sum = 0.f;
    for (int c = tid; c < cols; c += blockSize) {
        float v = expf(x[r * cols + c] - row_max);
        y[r * cols + c] = v;
        thread_sum += v;
    }

    __shared__ float shared_sum;
    if (tid == 0) shared_sum = 0.f;
    __syncthreads();

    atomicAdd(&shared_sum, thread_sum);
    __syncthreads();

    float row_sum = shared_sum;

    for (int c = tid; c < cols; c += blockSize) {
        y[r * cols + c] /= row_sum;
    }
}

// ============ Kernel 3: Warp-Level Reduction (Fastest) ============
__inline__ __device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void SoftmaxWarpKernel(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;

    int tid = threadIdx.x;
    int lane = tid & 31;

    float val = (tid < cols) ? x[r * cols + tid] : -INFINITY;
    float max_val = warpReduceMax(val);
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    val = (tid < cols) ? expf(val - max_val) : 0.f;
    float sum_val = warpReduceSum(val);
    sum_val = __shfl_sync(0xffffffff, sum_val, 0);

    if (tid < cols) {
        y[r * cols + tid] = val / sum_val;
    }
}

// ============ Kernel 4: Vectorized + WarpLevel ============
__global__ void SoftmaxVectorizedKernel(const float* __restrict__ x,
                                         float* __restrict__ y,
                                         int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int vector_len = 4;
    int col_base = lane * vector_len;

    float4 val4;
    if (col_base + 3 < cols) {
        *reinterpret_cast<float4*>(&val4) = reinterpret_cast<const float4*>(&x[r * cols + col_base])[0];
    } else {
        val4.x = (col_base < cols) ? x[r * cols + col_base] : -INFINITY;
        val4.y = (col_base + 1 < cols) ? x[r * cols + col_base + 1] : -INFINITY;
        val4.z = (col_base + 2 < cols) ? x[r * cols + col_base + 2] : -INFINITY;
        val4.w = (col_base + 3 < cols) ? x[r * cols + col_base + 3] : -INFINITY;
    }

    float max_val = fmaxf(fmaxf(val4.x, val4.y), fmaxf(val4.z, val4.w));
    max_val = warpReduceMax(max_val);
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    val4.x = expf(val4.x - max_val);
    val4.y = expf(val4.y - max_val);
    val4.z = expf(val4.z - max_val);
    val4.w = expf(val4.w - max_val);

    float sum_val = val4.x + val4.y + val4.z + val4.w;
    sum_val = warpReduceSum(sum_val);
    sum_val = __shfl_sync(0xffffffff, sum_val, 0);

    if (col_base + 3 < cols) {
        val4.x /= sum_val;
        val4.y /= sum_val;
        val4.z /= sum_val;
        val4.w /= sum_val;
        reinterpret_cast<float4*>(&y[r * cols + col_base])[0] = val4;
    } else {
        if (col_base < cols) y[r * cols + col_base] = val4.x / sum_val;
        if (col_base + 1 < cols) y[r * cols + col_base + 1] = val4.y / sum_val;
        if (col_base + 2 < cols) y[r * cols + col_base + 2] = val4.z / sum_val;
    }
}

// ============ CUB-based Softmax ============
struct MaxFunctor {
    __device__ __forceinline__ float operator()(const float& a, const float& b) const {
        return fmaxf(a, b);
    }
};

struct SumFunctor {
    __device__ __forceinline__ float operator()(const float& a, const float& b) const {
        return a + b;
    }
};

__global__ void ExpShiftKernel(const float* __restrict__ x,
                               float* __restrict__ e,
                               int cols,
                               float maxv) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < cols) e[c] = expf(x[c] - maxv);
}

__global__ void NormalizeKernel(const float* __restrict__ e,
                                float* __restrict__ y,
                                int cols,
                                float inv_sum) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < cols) y[c] = e[c] * inv_sum;
}

static double RunCUBSoftmax(const float* h_x, float* h_y, int rows, int cols) {
    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, rows * cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, rows * cols * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(s));

    for (int r = 0; r < rows; ++r) {
        const float* row_x = d_x + r * cols;
        float* row_y = d_y + r * cols;

        float *d_max = nullptr, *d_sum = nullptr, *d_exp = nullptr;
        void *tmp_max = nullptr, *tmp_sum = nullptr;
        size_t max_bytes = 0, sum_bytes = 0;

        CHECK_CUDA(cudaMalloc(&d_max, sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_sum, sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_exp, cols * sizeof(float)));

        cub::DeviceReduce::Max(nullptr, max_bytes, row_x, d_max, cols);
        CHECK_CUDA(cudaMalloc(&tmp_max, max_bytes));
        cub::DeviceReduce::Max(tmp_max, max_bytes, row_x, d_max, cols);

        float h_max = 0.f;
        CHECK_CUDA(cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));

        const int threads = 256;
        const int blocks = (cols + threads - 1) / threads;
        ExpShiftKernel<<<blocks, threads>>>(row_x, d_exp, cols, h_max);

        cub::DeviceReduce::Sum(nullptr, sum_bytes, d_exp, d_sum, cols);
        CHECK_CUDA(cudaMalloc(&tmp_sum, sum_bytes));
        cub::DeviceReduce::Sum(tmp_sum, sum_bytes, d_exp, d_sum, cols);

        float h_sum = 0.f;
        CHECK_CUDA(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
        NormalizeKernel<<<blocks, threads>>>(d_exp, row_y, cols, 1.0f / h_sum);

        CHECK_CUDA(cudaFree(tmp_max));
        CHECK_CUDA(cudaFree(tmp_sum));
        CHECK_CUDA(cudaFree(d_max));
        CHECK_CUDA(cudaFree(d_sum));
        CHECK_CUDA(cudaFree(d_exp));
    }

    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));

    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));

    CHECK_CUDA(cudaMemcpy(h_y, d_y, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    return ms;
}

// ============ cuDNN Softmax ============
static double RunCuDNNSoftmax(cudnnHandle_t handle,
                                const float* h_x, float* h_y,
                                int rows, int cols) {
    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, rows * cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, rows * cols * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    cudnnTensorDescriptor_t x_desc, y_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, rows, cols));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, rows, cols));

    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(s));

    float alpha = 1.f, beta = 0.f;
    CHECK_CUDNN(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                     &alpha, x_desc, d_x, &beta, y_desc, d_y));

    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));

    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));

    CHECK_CUDA(cudaMemcpy(h_y, d_y, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(x_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(y_desc));
    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    return ms;
}

// ============ CPU Reference ============
static void SoftmaxCPU(const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float maxv = x[r * cols];
        for (int c = 1; c < cols; ++c) {
            maxv = std::max(maxv, x[r * cols + c]);
        }
        float sum = 0.f;
        for (int c = 0; c < cols; ++c) {
            float v = std::exp(x[r * cols + c] - maxv);
            y[r * cols + c] = v;
            sum += v;
        }
        for (int c = 0; c < cols; ++c) {
            y[r * cols + c] /= sum;
        }
    }
}

// ============ Benchmark Helpers ============
template<typename KernelFunc>
double RunKernel(KernelFunc kernel, const float* h_x, float* h_y,
                  int rows, int cols, int iterations, size_t smem_bytes = 0) {
    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, rows * cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, rows * cols * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    kernel<<<rows, 256, smem_bytes>>>(d_x, d_y, rows, cols);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<double> times;
    for (int iter = 0; iter < iterations; ++iter) {
        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));

        CHECK_CUDA(cudaEventRecord(s));
        kernel<<<rows, 256, smem_bytes>>>(d_x, d_y, rows, cols);
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));

        float ms = 0.f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
        times.push_back(ms);

        CHECK_CUDA(cudaEventDestroy(s));
        CHECK_CUDA(cudaEventDestroy(e));
    }

    CHECK_CUDA(cudaMemcpy(h_y, d_y, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    std::sort(times.begin(), times.end());
    double sum = 0;
    for (size_t i = 1; i + 1 < times.size(); ++i) {
        sum += times[i];
    }
    return sum / (times.size() - 2);
}

// ============ Main ============
int main() {
    constexpr int ITERATIONS = 10;

    std::cout << "========================================\n";
    std::cout << "  Softmax Performance Comparison\n";
    std::cout << "  Naive | SharedMem | WarpReduce | Vectorized | cuDNN\n";
    std::cout << "========================================\n\n";

    std::vector<std::pair<int, int>> test_cases = {
        {64, 512},
        {64, 1024},
        {64, 4096},
        {128, 512},
        {128, 1024},
        {128, 4096},
        {256, 512},
        {256, 1024},
        {256, 4096},
        {512, 768},
        {512, 1024},
        {512, 4096},
    };

    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/softmax_all_comparison.csv");
    ofs << "rows,cols,naive_ms,sharedmem_ms,warp_ms,vectorized_ms,cudnn_ms";
    ofs << ",naive_gflops,sharedmem_gflops,warp_gflops,vectorized_gflops,cudnn_gflops\n";

    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    std::cout << std::left
              << std::setw(6) << "Rows"
              << std::setw(6) << "Cols"
              << std::setw(12) << "Naive"
              << std::setw(12) << "SharedMem"
              << std::setw(12) << "WarpReduce"
              << std::setw(12) << "Vectorized"
              << std::setw(12) << "cuDNN"
              << "\n";
    std::cout << std::string(66, '-') << "\n";

    for (const auto& [rows, cols] : test_cases) {
        std::vector<float> h_x(rows * cols), h_cpu(rows * cols), h_gpu(rows * cols);

        common::InitMatrix(h_x, rows, cols);

        auto t0 = std::chrono::high_resolution_clock::now();
        SoftmaxCPU(h_x.data(), h_cpu.data(), rows, cols);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << std::left
                  << std::setw(6) << rows
                  << std::setw(6) << cols;

        double naive_ms = RunKernel(SoftmaxNaiveKernel, h_x.data(), h_gpu.data(), rows, cols, ITERATIONS);
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(12) << naive_ms;

        double sharedmem_ms = RunKernel(SoftmaxSharedMemKernel, h_x.data(), h_gpu.data(),
                                        rows, cols, ITERATIONS, 256 * sizeof(float));
        std::cout << std::setw(12) << sharedmem_ms;

        double warp_ms = RunKernel(SoftmaxWarpKernel, h_x.data(), h_gpu.data(), rows, cols, ITERATIONS);
        std::cout << std::setw(12) << warp_ms;

        double vectorized_ms = RunKernel(SoftmaxVectorizedKernel, h_x.data(), h_gpu.data(), rows, cols, ITERATIONS);
        std::cout << std::setw(12) << vectorized_ms;

        double cudnn_ms = RunCuDNNSoftmax(cudnn, h_x.data(), h_gpu.data(), rows, cols);
        std::cout << std::setw(12) << cudnn_ms << "\n";

        double gflops = 2.0 * rows * cols / 1e6;
        ofs << rows << "," << cols << ","
            << naive_ms << "," << sharedmem_ms << "," << warp_ms << "," << vectorized_ms << "," << cudnn_ms << ","
            << gflops / naive_ms << "," << gflops / sharedmem_ms << "," << gflops / warp_ms << ","
            << gflops / vectorized_ms << "," << gflops / cudnn_ms << "\n";
    }

    CHECK_CUDNN(cudnnDestroy(cudnn));

    std::cout << "\n========================================\n";
    std::cout << "Notes:\n";
    std::cout << "  - WarpReduce: Uses __shfl_sync for warp-level reduction\n";
    std::cout << "  - Vectorized: Uses float4 for vectorized memory access\n";
    std::cout << "  - cuDNN: NVIDIA's highly optimized DNN library\n";
    std::cout << "  - Softmax: exp(x_i) / sum(exp(x_j)) with numerically stable max subtraction\n";
    std::cout << "========================================\n";

    return 0;
}
