// Softmax Performance Comparison: V0/V1/V2/V3 + CUB/cuBLAS/CUTLASS/cuDNN

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
    }                                                                             \
  } while (0)

#define CHECK_CUBLAS(call)                                                        \
  do {                                                                            \
    cublasStatus_t s__ = (call);                                                  \
    if (s__ != CUBLAS_STATUS_SUCCESS) {                                           \
      std::cerr << "cuBLAS error: " << static_cast<int>(s__) << std::endl;       \
      std::exit(EXIT_FAILURE);                                                    \
    }                                                                             \
  } while (0)

#if __has_include(<cutlass/cutlass.h>)
#define HAS_CUTLASS 1
#include <cutlass/cutlass.h>
#else
#define HAS_CUTLASS 0
#endif

static constexpr int kWarpSize = 32;

// ============================================================================
// V0
// ============================================================================
__global__ void SoftmaxV0NaiveKernel(const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int rows, int cols) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;

  float maxv = x[r * cols];
  for (int c = 1; c < cols; ++c) maxv = fmaxf(maxv, x[r * cols + c]);

  float sum = 0.f;
  for (int c = 0; c < cols; ++c) {
    float v = expf(x[r * cols + c] - maxv);
    y[r * cols + c] = v;
    sum += v;
  }
  for (int c = 0; c < cols; ++c) y[r * cols + c] /= sum;
}

// ============================================================================
// V1
// ============================================================================
__global__ void SoftmaxV1SharedMemKernel(const float* __restrict__ x,
                                         float* __restrict__ y,
                                         int rows, int cols) {
  extern __shared__ float sdata[];
  int r = blockIdx.x;
  int tid = threadIdx.x;
  int block_size = blockDim.x;
  if (r >= rows) return;

  float thread_max = -INFINITY;
  for (int c = tid; c < cols; c += block_size) {
    thread_max = fmaxf(thread_max, x[r * cols + c]);
  }
  sdata[tid] = thread_max;
  __syncthreads();

  for (int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  float row_max = sdata[0];

  float thread_sum = 0.f;
  for (int c = tid; c < cols; c += block_size) {
    float v = expf(x[r * cols + c] - row_max);
    y[r * cols + c] = v;
    thread_sum += v;
  }

  __shared__ float row_sum;
  if (tid == 0) row_sum = 0.f;
  __syncthreads();
  atomicAdd(&row_sum, thread_sum);
  __syncthreads();

  for (int c = tid; c < cols; c += block_size) {
    y[r * cols + c] /= row_sum;
  }
}

// ============================================================================
// V2
// ============================================================================
__device__ __forceinline__ float WarpReduceMax(float val) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

__device__ __forceinline__ float WarpReduceSum(float val) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__global__ void SoftmaxV2WarpKernel(const float* __restrict__ x,
                                    float* __restrict__ y,
                                    int rows, int cols) {
  int r = blockIdx.x;
  int tid = threadIdx.x;
  if (r >= rows) return;

  float v = (tid < cols) ? x[r * cols + tid] : -INFINITY;
  float maxv = WarpReduceMax(v);
  maxv = __shfl_sync(0xffffffff, maxv, 0);

  v = (tid < cols) ? expf(v - maxv) : 0.f;
  float sumv = WarpReduceSum(v);
  sumv = __shfl_sync(0xffffffff, sumv, 0);

  if (tid < cols) y[r * cols + tid] = v / sumv;
}

// ============================================================================
// V3
// ============================================================================
__global__ void SoftmaxV3VectorizedKernel(const float* __restrict__ x,
                                          float* __restrict__ y,
                                          int rows, int cols) {
  int r = blockIdx.x;
  int tid = threadIdx.x;
  int lane = tid & 31;
  if (r >= rows) return;

  int base = lane * 4;
  float4 v4;
  if (base + 3 < cols) {
    v4 = reinterpret_cast<const float4*>(x + r * cols + base)[0];
  } else {
    v4.x = (base < cols) ? x[r * cols + base] : -INFINITY;
    v4.y = (base + 1 < cols) ? x[r * cols + base + 1] : -INFINITY;
    v4.z = (base + 2 < cols) ? x[r * cols + base + 2] : -INFINITY;
    v4.w = (base + 3 < cols) ? x[r * cols + base + 3] : -INFINITY;
  }

  float maxv = fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w));
  maxv = WarpReduceMax(maxv);
  maxv = __shfl_sync(0xffffffff, maxv, 0);

  v4.x = expf(v4.x - maxv);
  v4.y = expf(v4.y - maxv);
  v4.z = expf(v4.z - maxv);
  v4.w = expf(v4.w - maxv);
  float sumv = WarpReduceSum(v4.x + v4.y + v4.z + v4.w);
  sumv = __shfl_sync(0xffffffff, sumv, 0);

  if (base + 3 < cols) {
    reinterpret_cast<float4*>(y + r * cols + base)[0] =
        make_float4(v4.x / sumv, v4.y / sumv, v4.z / sumv, v4.w / sumv);
  } else {
    if (base < cols) y[r * cols + base] = v4.x / sumv;
    if (base + 1 < cols) y[r * cols + base + 1] = v4.y / sumv;
    if (base + 2 < cols) y[r * cols + base + 2] = v4.z / sumv;
    if (base + 3 < cols) y[r * cols + base + 3] = v4.w / sumv;
  }
}

// ============================================================================
// CUB baseline
// ============================================================================
__global__ void ExpShiftKernel(const float* __restrict__ x,
                               float* __restrict__ e,
                               int cols, float maxv) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < cols) e[c] = expf(x[c] - maxv);
}

__global__ void NormalizeKernel(const float* __restrict__ e,
                                float* __restrict__ y,
                                int cols, float inv_sum) {
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
    int threads = 256;
    int blocks = (cols + threads - 1) / threads;
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

// ============================================================================
// cuBLAS-assisted baseline
// ============================================================================
__global__ void RowMaxKernel(const float* __restrict__ x,
                             float* __restrict__ row_max,
                             int rows, int cols) {
  int r = blockIdx.x;
  int tid = threadIdx.x;
  if (r >= rows) return;
  extern __shared__ float smax[];
  float v = -INFINITY;
  for (int c = tid; c < cols; c += blockDim.x) v = fmaxf(v, x[r * cols + c]);
  smax[tid] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) smax[tid] = fmaxf(smax[tid], smax[tid + s]);
    __syncthreads();
  }
  if (tid == 0) row_max[r] = smax[0];
}

__global__ void ExpShiftRowsKernel(const float* __restrict__ x,
                                   const float* __restrict__ row_max,
                                   float* __restrict__ e,
                                   int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = rows * cols;
  if (idx >= n) return;
  int r = idx / cols;
  e[idx] = expf(x[idx] - row_max[r]);
}

__global__ void NormalizeRowsKernel(const float* __restrict__ e,
                                    const float* __restrict__ row_sum,
                                    float* __restrict__ y,
                                    int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = rows * cols;
  if (idx >= n) return;
  int r = idx / cols;
  y[idx] = e[idx] / row_sum[r];
}

static double RunCuBLASSoftmax(const float* h_x, float* h_y, int rows, int cols) {
  int n = rows * cols;
  float *d_x, *d_y, *d_e, *d_row_max, *d_row_sum;
  CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_y, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_e, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_row_max, rows * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_row_sum, rows * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  int threads = 256;
  int blocks_rows = rows;
  int blocks_all = (n + threads - 1) / threads;

  cudaEvent_t s, e;
  CHECK_CUDA(cudaEventCreate(&s));
  CHECK_CUDA(cudaEventCreate(&e));
  CHECK_CUDA(cudaEventRecord(s));

  RowMaxKernel<<<blocks_rows, threads, threads * sizeof(float)>>>(d_x, d_row_max, rows, cols);
  ExpShiftRowsKernel<<<blocks_all, threads>>>(d_x, d_row_max, d_e, rows, cols);
  for (int r = 0; r < rows; ++r) {
    float sum = 0.f;
    CHECK_CUBLAS(cublasSasum(handle, cols, d_e + r * cols, 1, &sum));
    CHECK_CUDA(cudaMemcpy(d_row_sum + r, &sum, sizeof(float), cudaMemcpyHostToDevice));
  }
  NormalizeRowsKernel<<<blocks_all, threads>>>(d_e, d_row_sum, d_y, rows, cols);

  CHECK_CUDA(cudaEventRecord(e));
  CHECK_CUDA(cudaEventSynchronize(e));
  float ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));

  CHECK_CUDA(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaEventDestroy(s));
  CHECK_CUDA(cudaEventDestroy(e));
  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));
  CHECK_CUDA(cudaFree(d_e));
  CHECK_CUDA(cudaFree(d_row_max));
  CHECK_CUDA(cudaFree(d_row_sum));
  return ms;
}

// ============================================================================
// CUTLASS hook
// ============================================================================
static double RunCUTLASSSoftmax(const float* h_x, float* h_y, int rows, int cols) {
  (void)h_x; (void)h_y; (void)rows; (void)cols;
#if HAS_CUTLASS
  // Placeholder: fill with real CUTLASS softmax pipeline when integrated.
  return -1.0;
#else
  return -1.0;
#endif
}

// ============================================================================
// cuDNN
// ============================================================================
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
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(
      x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, rows, cols));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(
      y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, rows, cols));

  cudaEvent_t s, e;
  CHECK_CUDA(cudaEventCreate(&s));
  CHECK_CUDA(cudaEventCreate(&e));
  CHECK_CUDA(cudaEventRecord(s));
  float alpha = 1.f, beta = 0.f;
  CHECK_CUDNN(cudnnSoftmaxForward(
      handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
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

// ============================================================================
// CPU reference
// ============================================================================
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

template <typename KernelFunc>
double RunKernel(KernelFunc kernel,
                 const float* h_x, float* h_y,
                 int rows, int cols,
                 int iterations,
                 int threads = 256,
                 size_t smem_bytes = 0) {
  float *d_x, *d_y;
  CHECK_CUDA(cudaMalloc(&d_x, rows * cols * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_y, rows * cols * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_x, h_x, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

  kernel<<<rows, threads, smem_bytes>>>(d_x, d_y, rows, cols);
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<double> times;
  for (int i = 0; i < iterations; ++i) {
    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(s));
    kernel<<<rows, threads, smem_bytes>>>(d_x, d_y, rows, cols);
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
  double sum = 0.0;
  for (size_t i = 1; i + 1 < times.size(); ++i) sum += times[i];
  return sum / static_cast<double>(times.size() - 2);
}

int main() {
  constexpr int kIterations = 10;
  std::vector<std::pair<int, int>> test_cases = {
      {64, 512}, {64, 1024}, {64, 4096},
      {128, 512}, {128, 1024}, {128, 4096},
      {256, 512}, {256, 1024}, {256, 4096},
      {512, 768}, {512, 1024}, {512, 4096},
  };

  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/softmax_all_comparison.csv");
  ofs << "rows,cols,v0_ms,v1_ms,v2_ms,v3_ms,cub_ms,cublas_ms,cutlass_ms,cudnn_ms,"
         "v0_gflops,v1_gflops,v2_gflops,v3_gflops,cub_gflops,cublas_gflops,cutlass_gflops,cudnn_gflops\n";

  cudnnHandle_t cudnn;
  CHECK_CUDNN(cudnnCreate(&cudnn));

  std::cout << std::left
            << std::setw(6) << "Rows"
            << std::setw(6) << "Cols"
            << std::setw(10) << "V0"
            << std::setw(10) << "V1"
            << std::setw(10) << "V2"
            << std::setw(10) << "V3"
            << std::setw(10) << "CUB"
            << std::setw(10) << "cuBLAS"
            << std::setw(10) << "CUTLASS"
            << std::setw(12) << "cuDNN"
            << "\n";
  std::cout << std::string(94, '-') << "\n";

  for (auto [rows, cols] : test_cases) {
    std::vector<float> h_x(rows * cols), h_cpu(rows * cols), h_gpu(rows * cols);
    common::InitMatrix(h_x, rows, cols);
    SoftmaxCPU(h_x.data(), h_cpu.data(), rows, cols);

    double v0_ms = RunKernel(SoftmaxV0NaiveKernel, h_x.data(), h_gpu.data(), rows, cols, kIterations, 256);
    double v1_ms = RunKernel(SoftmaxV1SharedMemKernel, h_x.data(), h_gpu.data(), rows, cols, kIterations, 256, 256 * sizeof(float));
    double v2_ms = RunKernel(SoftmaxV2WarpKernel, h_x.data(), h_gpu.data(), rows, cols, kIterations, 32);
    double v3_ms = RunKernel(SoftmaxV3VectorizedKernel, h_x.data(), h_gpu.data(), rows, cols, kIterations, 32);
    double cub_ms = RunCUBSoftmax(h_x.data(), h_gpu.data(), rows, cols);
    double cublas_ms = RunCuBLASSoftmax(h_x.data(), h_gpu.data(), rows, cols);
    double cutlass_ms = RunCUTLASSSoftmax(h_x.data(), h_gpu.data(), rows, cols);
    double cudnn_ms = RunCuDNNSoftmax(cudnn, h_x.data(), h_gpu.data(), rows, cols);

    std::cout << std::fixed << std::setprecision(3)
              << std::setw(6) << rows
              << std::setw(6) << cols
              << std::setw(10) << v0_ms
              << std::setw(10) << v1_ms
              << std::setw(10) << v2_ms
              << std::setw(10) << v3_ms
              << std::setw(10) << cub_ms
              << std::setw(10) << cublas_ms
              << std::setw(10) << cutlass_ms
              << std::setw(12) << cudnn_ms
              << "\n";

    const double gflops = 2.0 * rows * cols / 1e6;
    ofs << rows << "," << cols << ","
        << v0_ms << "," << v1_ms << "," << v2_ms << "," << v3_ms << ","
        << cub_ms << "," << cublas_ms << "," << cutlass_ms << "," << cudnn_ms << ","
        << gflops / v0_ms << "," << gflops / v1_ms << "," << gflops / v2_ms << "," << gflops / v3_ms << ","
        << gflops / cub_ms << "," << gflops / cublas_ms << ","
        << (cutlass_ms > 0 ? gflops / cutlass_ms : -1.0) << ","
        << gflops / cudnn_ms << "\n";
  }

  CHECK_CUDNN(cudnnDestroy(cudnn));
  return 0;
}
