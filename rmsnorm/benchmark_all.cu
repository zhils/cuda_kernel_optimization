// RMSNorm Performance Comparison: V0 / V1 / V2 / V3 / CUB / cuDNN

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <cudnn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

#define CHECK_CUDNN(call)                                                        \
  do {                                                                           \
    cudnnStatus_t s__ = (call);                                                  \
    if (s__ != CUDNN_STATUS_SUCCESS) {                                           \
      std::cerr << "cuDNN error: " << s__ << std::endl;                          \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#define CHECK_CUBLAS(call)                                                       \
  do {                                                                           \
    cublasStatus_t s__ = (call);                                                 \
    if (s__ != CUBLAS_STATUS_SUCCESS) {                                          \
      std::cerr << "cuBLAS error: " << static_cast<int>(s__) << std::endl;      \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

namespace {
static constexpr float kEps = 1e-5f;
static constexpr int kWarpSize = 32;
static constexpr int kThreadsV1V2 = 256;
static constexpr int kMaxThreadsV3 = 512;
static constexpr std::size_t kMaxDynamicSmemBytes = 40 * 1024;
}  // namespace

#if __has_include(<cutlass/cutlass.h>)
#define HAS_CUTLASS 1
#include <cutlass/cutlass.h>
#else
#define HAS_CUTLASS 0
#endif

// ============================================================================
// Shared helpers
// ============================================================================
__device__ __forceinline__ float4 LoadFloat4(const float* p) {
  return __ldg(reinterpret_cast<const float4*>(p));
}

__device__ __forceinline__ void StoreFloat4(float* p, const float4& v) {
  *reinterpret_cast<float4*>(p) = v;
}

__device__ __forceinline__ float WarpReduceSum(float val) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float BlockReduceSumV1V2(float val) {
  __shared__ float warp_sums[kThreadsV1V2 / kWarpSize];
  const int tid = threadIdx.x;
  const int lane = tid % kWarpSize;
  const int warp_id = tid / kWarpSize;
  const int warp_count = (blockDim.x + kWarpSize - 1) / kWarpSize;

  val = WarpReduceSum(val);
  if (lane == 0) warp_sums[warp_id] = val;
  __syncthreads();

  float sum = 0.f;
  if (warp_id == 0) {
    sum = (lane < warp_count) ? warp_sums[lane] : 0.f;
    sum = WarpReduceSum(sum);
  }
  __syncthreads();
  return sum;
}

__device__ __forceinline__ float BlockReduceSumV3(float val) {
  __shared__ float warp_sums[kMaxThreadsV3 / kWarpSize];
  const int tid = threadIdx.x;
  const int lane = tid % kWarpSize;
  const int warp_id = tid / kWarpSize;
  const int warp_count = (blockDim.x + kWarpSize - 1) / kWarpSize;

  val = WarpReduceSum(val);
  if (lane == 0) warp_sums[warp_id] = val;
  __syncthreads();

  float sum = 0.f;
  if (warp_id == 0) {
    sum = (lane < warp_count) ? warp_sums[lane] : 0.f;
    sum = WarpReduceSum(sum);
  }
  __syncthreads();
  return sum;
}

inline bool CanStageRowV1V2(int cols) {
  const std::size_t need =
      (static_cast<std::size_t>(cols) + static_cast<std::size_t>(kThreadsV1V2)) * sizeof(float);
  return need <= kMaxDynamicSmemBytes;
}

inline int PickThreadsByColsV3(int cols) {
  if (cols <= 128) return 64;
  if (cols <= 512) return 128;
  if (cols <= 2048) return 256;
  return 512;
}

inline bool CanStageRowV3(int cols, int threads) {
  const std::size_t need =
      (static_cast<std::size_t>(cols) + static_cast<std::size_t>(threads)) * sizeof(float);
  return need <= kMaxDynamicSmemBytes;
}

template <typename LaunchFn>
double MeasureKernelMs(LaunchFn launch, int iterations = 10) {
  for (int i = 0; i < 3; ++i) launch();
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> times;
  times.reserve(iterations);
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < iterations; ++i) {
    CHECK_CUDA(cudaEventRecord(start));
    launch();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    times.push_back(ms);
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  std::sort(times.begin(), times.end());
  if (times.size() > 2) {
    double sum = 0.0;
    for (size_t i = 1; i + 1 < times.size(); ++i) sum += times[i];
    return sum / static_cast<double>(times.size() - 2);
  }
  double sum = 0.0;
  for (float t : times) sum += t;
  return sum / static_cast<double>(times.size());
}

static void RMSNormCPU(const float* x, float* y, const float* weight,
                       int rows, int cols, float eps) {
  for (int r = 0; r < rows; ++r) {
    float sq_sum = 0.f;
    for (int c = 0; c < cols; ++c) {
      const float val = x[r * cols + c];
      sq_sum += val * val;
    }
    const float rms = 1.f / sqrtf(sq_sum / cols + eps);
    for (int c = 0; c < cols; ++c) {
      y[r * cols + c] = x[r * cols + c] * rms * weight[c];
    }
  }
}

__global__ void ElementwiseSquareKernel(const float* __restrict__ x,
                                        float* __restrict__ x2,
                                        int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) x2[idx] = x[idx] * x[idx];
}

__global__ void RMSNormScaleFromSqSumKernel(const float* __restrict__ x,
                                            float* __restrict__ y,
                                            const float* __restrict__ weight,
                                            const float* __restrict__ sq_sums,
                                            int rows, int cols, float eps) {
  int tid = threadIdx.x;
  int r = blockIdx.x;
  if (r >= rows) return;
  int offset = r * cols;
  float rms = rsqrtf(sq_sums[r] / cols + eps);
  for (int c = tid; c < cols; c += blockDim.x) {
    y[offset + c] = x[offset + c] * rms * weight[c];
  }
}

// ============================================================================
// V0: Naive (same as rmsnorm_v0_naive.cu)
// ============================================================================
__global__ void RMSNormV0Kernel(const float* x, float* y, const float* weight,
                                int rows, int cols, float eps) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float sq_sum = 0.f;
  const int offset = r * cols;
  for (int c = 0; c < cols; ++c) {
    float val = x[offset + c];
    sq_sum += val * val;
  }
  const float rms = rsqrtf(sq_sum / cols + eps);
  for (int c = 0; c < cols; ++c) {
    y[offset + c] = x[offset + c] * rms * weight[c];
  }
}

// ============================================================================
// V1: staged + stream (same strategy as rmsnorm_v1.cu)
// ============================================================================
__global__ void RMSNormV1StagedKernel(const float* __restrict__ x, float* __restrict__ y,
                                      const float* __restrict__ weight,
                                      int rows, int cols, float eps) {
  extern __shared__ float sdata[];
  float* s_row = sdata;
  float* s_red = sdata + cols;

  const int tid = threadIdx.x;
  const int r = blockIdx.x;
  if (r >= rows) return;

  const float* row_x = x + static_cast<std::size_t>(r) * cols;
  float* row_y = y + static_cast<std::size_t>(r) * cols;
  const bool align4 =
      ((reinterpret_cast<std::uintptr_t>(row_x) % 16u) == 0u) && ((cols % 4) == 0);

  if (align4) {
    int c = tid * 4;
    for (; c + 3 < cols; c += kThreadsV1V2 * 4) {
      StoreFloat4(s_row + c, LoadFloat4(row_x + c));
    }
    for (int k = c; k < cols; ++k) s_row[k] = __ldg(row_x + k);
  } else {
    for (int k = tid; k < cols; k += kThreadsV1V2) s_row[k] = __ldg(row_x + k);
  }
  __syncthreads();

  float ps = 0.f;
  if (align4) {
    int c = tid * 4;
    for (; c + 3 < cols; c += kThreadsV1V2 * 4) {
      const float4 v = *reinterpret_cast<const float4*>(s_row + c);
      ps += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (int k = c; k < cols; ++k) {
      const float v = s_row[k];
      ps += v * v;
    }
  } else {
    for (int k = tid; k < cols; k += kThreadsV1V2) {
      const float v = s_row[k];
      ps += v * v;
    }
  }
  s_red[tid] = ps;
  __syncthreads();
  for (int s = kThreadsV1V2 / 2; s > 0; s >>= 1) {
    if (tid < s) s_red[tid] += s_red[tid + s];
    __syncthreads();
  }
  if (tid == 0) s_red[0] = rsqrtf(s_red[0] / static_cast<float>(cols) + eps);
  __syncthreads();
  const float rms = s_red[0];

  if (align4) {
    int c = tid * 4;
    for (; c + 3 < cols; c += kThreadsV1V2 * 4) {
      const float4 vx = *reinterpret_cast<const float4*>(s_row + c);
      const float4 vw = LoadFloat4(weight + c);
      StoreFloat4(row_y + c, make_float4(vx.x * rms * vw.x, vx.y * rms * vw.y,
                                          vx.z * rms * vw.z, vx.w * rms * vw.w));
    }
    for (int k = c; k < cols; ++k) row_y[k] = s_row[k] * rms * __ldg(weight + k);
  } else {
    for (int k = tid; k < cols; k += kThreadsV1V2) row_y[k] = s_row[k] * rms * __ldg(weight + k);
  }
}

__global__ void RMSNormV1StreamKernel(const float* __restrict__ x, float* __restrict__ y,
                                      const float* __restrict__ weight,
                                      int rows, int cols, float eps) {
  __shared__ float s_red[kThreadsV1V2];
  const int tid = threadIdx.x;
  const int r = blockIdx.x;
  if (r >= rows) return;

  const float* row_x = x + static_cast<std::size_t>(r) * cols;
  float* row_y = y + static_cast<std::size_t>(r) * cols;
  const bool align4 =
      ((reinterpret_cast<std::uintptr_t>(row_x) % 16u) == 0u) && ((cols % 4) == 0);

  float ps = 0.f;
  if (align4) {
    int c = tid * 4;
    for (; c + 3 < cols; c += kThreadsV1V2 * 4) {
      const float4 v = LoadFloat4(row_x + c);
      ps += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (int k = c; k < cols; ++k) {
      const float v = __ldg(row_x + k);
      ps += v * v;
    }
  } else {
    for (int k = tid; k < cols; k += kThreadsV1V2) {
      const float v = __ldg(row_x + k);
      ps += v * v;
    }
  }
  s_red[tid] = ps;
  __syncthreads();
  for (int s = kThreadsV1V2 / 2; s > 0; s >>= 1) {
    if (tid < s) s_red[tid] += s_red[tid + s];
    __syncthreads();
  }
  if (tid == 0) s_red[0] = rsqrtf(s_red[0] / static_cast<float>(cols) + eps);
  __syncthreads();
  const float rms = s_red[0];

  if (align4) {
    int c = tid * 4;
    for (; c + 3 < cols; c += kThreadsV1V2 * 4) {
      const float4 vx = LoadFloat4(row_x + c);
      const float4 vw = LoadFloat4(weight + c);
      StoreFloat4(row_y + c, make_float4(vx.x * rms * vw.x, vx.y * rms * vw.y,
                                          vx.z * rms * vw.z, vx.w * rms * vw.w));
    }
    for (int k = c; k < cols; ++k) row_y[k] = __ldg(row_x + k) * rms * __ldg(weight + k);
  } else {
    for (int k = tid; k < cols; k += kThreadsV1V2) row_y[k] = __ldg(row_x + k) * rms * __ldg(weight + k);
  }
}

// ============================================================================
// V2: V1 + warp/block reduction path (same strategy as rmsnorm_v2.cu)
// ============================================================================
__global__ void RMSNormV2StagedKernel(const float* __restrict__ x, float* __restrict__ y,
                                      const float* __restrict__ weight,
                                      int rows, int cols, float eps) {
  extern __shared__ float sdata[];
  float* s_row = sdata;
  float* s_red = sdata + cols;

  const int tid = threadIdx.x;
  const int r = blockIdx.x;
  if (r >= rows) return;

  const int offset = r * cols;
  const float* row_x = x + static_cast<std::size_t>(offset);
  float* row_y = y + static_cast<std::size_t>(offset);
  const bool align4 =
      (cols % 4 == 0) &&
      (reinterpret_cast<std::uintptr_t>(row_x) % 16u == 0u) &&
      (reinterpret_cast<std::uintptr_t>(row_y) % 16u == 0u) &&
      (reinterpret_cast<std::uintptr_t>(weight) % 16u == 0u);

  if (align4) {
    int c = tid * 4;
    for (; c + 3 < cols; c += blockDim.x * 4) StoreFloat4(s_row + c, LoadFloat4(row_x + c));
    for (int c1 = c; c1 < cols; ++c1) s_row[c1] = __ldg(row_x + c1);
  } else {
    for (int c = tid; c < cols; c += blockDim.x) s_row[c] = __ldg(row_x + c);
  }
  __syncthreads();

  float sq_sum = 0.f;
  if (align4) {
    int c = tid * 4;
    for (; c + 3 < cols; c += blockDim.x * 4) {
      const float4 v = *reinterpret_cast<const float4*>(s_row + c);
      sq_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (int c1 = c; c1 < cols; ++c1) {
      const float v = s_row[c1];
      sq_sum += v * v;
    }
  } else {
    for (int c = tid; c < cols; c += blockDim.x) {
      const float v = s_row[c];
      sq_sum += v * v;
    }
  }

  sq_sum = BlockReduceSumV1V2(sq_sum);
  if (tid == 0) s_red[0] = rsqrtf(sq_sum / static_cast<float>(cols) + eps);
  __syncthreads();
  const float rms = s_red[0];

  if (align4) {
    int c = tid * 4;
    for (; c + 3 < cols; c += blockDim.x * 4) {
      const float4 vx = *reinterpret_cast<const float4*>(s_row + c);
      const float4 vw = LoadFloat4(weight + c);
      StoreFloat4(row_y + c, make_float4(vx.x * rms * vw.x, vx.y * rms * vw.y,
                                          vx.z * rms * vw.z, vx.w * rms * vw.w));
    }
    for (int c1 = c; c1 < cols; ++c1) row_y[c1] = s_row[c1] * rms * __ldg(weight + c1);
  } else {
    for (int c = tid; c < cols; c += blockDim.x) row_y[c] = s_row[c] * rms * __ldg(weight + c);
  }
}

__global__ void RMSNormV2StreamKernel(const float* __restrict__ x, float* __restrict__ y,
                                      const float* __restrict__ weight,
                                      int rows, int cols, float eps) {
  __shared__ float s_red[kThreadsV1V2 / kWarpSize];
  const int tid = threadIdx.x;
  const int r = blockIdx.x;
  if (r >= rows) return;

  const int offset = r * cols;
  const float* row_x = x + static_cast<std::size_t>(offset);
  float* row_y = y + static_cast<std::size_t>(offset);
  const int n4 = cols / 4;

  const bool align4 =
      (cols % 4 == 0) &&
      (reinterpret_cast<std::uintptr_t>(row_x) % 16u == 0u) &&
      (reinterpret_cast<std::uintptr_t>(row_y) % 16u == 0u) &&
      (reinterpret_cast<std::uintptr_t>(weight) % 16u == 0u);

  float sq_sum = 0.f;
  if (align4) {
    for (int c = tid; c < n4; c += blockDim.x) {
      const float4 v = LoadFloat4(row_x + c * 4);
      sq_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
  } else {
    for (int c = tid; c < cols; c += blockDim.x) {
      const float v = __ldg(row_x + c);
      sq_sum += v * v;
    }
  }

  sq_sum = BlockReduceSumV1V2(sq_sum);
  if (tid == 0) s_red[0] = rsqrtf(sq_sum / static_cast<float>(cols) + eps);
  __syncthreads();
  const float rms = s_red[0];

  if (align4) {
    for (int c = tid; c < n4; c += blockDim.x) {
      const float4 vx = LoadFloat4(row_x + c * 4);
      const float4 vw = LoadFloat4(weight + c * 4);
      StoreFloat4(row_y + c * 4, make_float4(vx.x * rms * vw.x, vx.y * rms * vw.y,
                                              vx.z * rms * vw.z, vx.w * rms * vw.w));
    }
  } else {
    for (int c = tid; c < cols; c += blockDim.x) row_y[c] = __ldg(row_x + c) * rms * __ldg(weight + c);
  }
}

// ============================================================================
// V3: V2 + auto block size by cols (same strategy as rmsnorm_v3.cu)
// ============================================================================
__global__ void RMSNormV3StagedKernel(const float* __restrict__ x, float* __restrict__ y,
                                      const float* __restrict__ weight,
                                      int rows, int cols, float eps) {
  extern __shared__ float sdata[];
  float* s_row = sdata;
  float* s_red = sdata + cols;

  const int tid = threadIdx.x;
  const int r = blockIdx.x;
  if (r >= rows) return;

  const int offset = r * cols;
  const float* row_x = x + static_cast<std::size_t>(offset);
  float* row_y = y + static_cast<std::size_t>(offset);
  const bool align4 =
      (cols % 4 == 0) &&
      (reinterpret_cast<std::uintptr_t>(row_x) % 16u == 0u) &&
      (reinterpret_cast<std::uintptr_t>(row_y) % 16u == 0u) &&
      (reinterpret_cast<std::uintptr_t>(weight) % 16u == 0u);

  if (align4) {
    int c = tid * 4;
    for (; c + 3 < cols; c += blockDim.x * 4) StoreFloat4(s_row + c, LoadFloat4(row_x + c));
    for (int c1 = c; c1 < cols; ++c1) s_row[c1] = __ldg(row_x + c1);
  } else {
    for (int c = tid; c < cols; c += blockDim.x) s_row[c] = __ldg(row_x + c);
  }
  __syncthreads();

  float sq_sum = 0.f;
  if (align4) {
    int c = tid * 4;
    for (; c + 3 < cols; c += blockDim.x * 4) {
      const float4 v = *reinterpret_cast<const float4*>(s_row + c);
      sq_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (int c1 = c; c1 < cols; ++c1) {
      const float v = s_row[c1];
      sq_sum += v * v;
    }
  } else {
    for (int c = tid; c < cols; c += blockDim.x) {
      const float v = s_row[c];
      sq_sum += v * v;
    }
  }

  sq_sum = BlockReduceSumV3(sq_sum);
  if (tid == 0) s_red[0] = rsqrtf(sq_sum / static_cast<float>(cols) + eps);
  __syncthreads();
  const float rms = s_red[0];

  if (align4) {
    int c = tid * 4;
    for (; c + 3 < cols; c += blockDim.x * 4) {
      const float4 vx = *reinterpret_cast<const float4*>(s_row + c);
      const float4 vw = LoadFloat4(weight + c);
      StoreFloat4(row_y + c, make_float4(vx.x * rms * vw.x, vx.y * rms * vw.y,
                                          vx.z * rms * vw.z, vx.w * rms * vw.w));
    }
    for (int c1 = c; c1 < cols; ++c1) row_y[c1] = s_row[c1] * rms * __ldg(weight + c1);
  } else {
    for (int c = tid; c < cols; c += blockDim.x) row_y[c] = s_row[c] * rms * __ldg(weight + c);
  }
}

__global__ void RMSNormV3StreamKernel(const float* __restrict__ x, float* __restrict__ y,
                                      const float* __restrict__ weight,
                                      int rows, int cols, float eps) {
  __shared__ float s_red[kMaxThreadsV3 / kWarpSize];
  const int tid = threadIdx.x;
  const int r = blockIdx.x;
  if (r >= rows) return;

  const int offset = r * cols;
  const float* row_x = x + static_cast<std::size_t>(offset);
  float* row_y = y + static_cast<std::size_t>(offset);
  const int n4 = cols / 4;

  const bool align4 =
      (cols % 4 == 0) &&
      (reinterpret_cast<std::uintptr_t>(row_x) % 16u == 0u) &&
      (reinterpret_cast<std::uintptr_t>(row_y) % 16u == 0u) &&
      (reinterpret_cast<std::uintptr_t>(weight) % 16u == 0u);

  float sq_sum = 0.f;
  if (align4) {
    for (int c = tid; c < n4; c += blockDim.x) {
      const float4 v = LoadFloat4(row_x + c * 4);
      sq_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
  } else {
    for (int c = tid; c < cols; c += blockDim.x) {
      const float v = __ldg(row_x + c);
      sq_sum += v * v;
    }
  }

  sq_sum = BlockReduceSumV3(sq_sum);
  if (tid == 0) s_red[0] = rsqrtf(sq_sum / static_cast<float>(cols) + eps);
  __syncthreads();
  const float rms = s_red[0];

  if (align4) {
    for (int c = tid; c < n4; c += blockDim.x) {
      const float4 vx = LoadFloat4(row_x + c * 4);
      const float4 vw = LoadFloat4(weight + c * 4);
      StoreFloat4(row_y + c * 4, make_float4(vx.x * rms * vw.x, vx.y * rms * vw.y,
                                              vx.z * rms * vw.z, vx.w * rms * vw.w));
    }
  } else {
    for (int c = tid; c < cols; c += blockDim.x) row_y[c] = __ldg(row_x + c) * rms * __ldg(weight + c);
  }
}

// ============================================================================
// CUB and cuDNN references
// ============================================================================
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
  if (tid == 0) sq_sum_out[r] = sq_sum;
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
  float rms = rsqrtf(sq_sums[r] / cols + eps);
  for (int c = tid; c < cols; c += blockDim.x) {
    y[offset + c] = x[offset + c] * rms * weight[c];
  }
}

static double RunCubRMSNorm(const std::vector<float>& h_x,
                            const std::vector<float>& h_weight,
                            std::vector<float>& h_y,
                            int rows, int cols) {
  int n = rows * cols;
  float *d_x, *d_y, *d_weight, *d_sq_sums;
  CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_y, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_weight, cols * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_sq_sums, rows * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_weight, h_weight.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = rows;
  double ms = MeasureKernelMs([&]() {
    RMSNormCubPrepKernel<<<blocks, threads>>>(d_x, d_sq_sums, rows, cols);
    RMSNormCubScaleKernel<<<blocks, threads>>>(d_x, d_y, d_weight, d_sq_sums, rows, cols, kEps);
  });

  CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));
  CHECK_CUDA(cudaFree(d_weight));
  CHECK_CUDA(cudaFree(d_sq_sums));
  return ms;
}

static double RunCudnnRMSNorm(const std::vector<float>& h_x,
                              const std::vector<float>& h_weight,
                              std::vector<float>& h_y,
                              int rows, int cols) {
  (void)h_x;
  (void)h_weight;
  (void)h_y;
  (void)rows;
  (void)cols;
  // cuDNN 9.x old API unavailable in this project path.
  return -1.0;
}

static double RunCuBLASRMSNorm(const std::vector<float>& h_x,
                               const std::vector<float>& h_weight,
                               std::vector<float>& h_y,
                               int rows, int cols) {
  const int n = rows * cols;
  float *d_x, *d_y, *d_w, *d_x2, *d_sq;
  CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_y, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_w, cols * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_x2, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_sq, rows * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_w, h_weight.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  const int threads = 256;
  const int blocks_n = (n + threads - 1) / threads;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // row-major x2[rows, cols] -> column-major view [cols, rows]
  // sq = x2 * ones(cols), use GEMV with transpose.
  std::vector<float> h_ones(cols, 1.0f);
  float* d_ones = nullptr;
  CHECK_CUDA(cudaMalloc(&d_ones, cols * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_ones, h_ones.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

  double ms = MeasureKernelMs([&]() {
    ElementwiseSquareKernel<<<blocks_n, threads>>>(d_x, d_x2, n);
    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T,
                             cols, rows,
                             &alpha,
                             d_x2, cols,
                             d_ones, 1,
                             &beta,
                             d_sq, 1));
    RMSNormScaleFromSqSumKernel<<<rows, threads>>>(d_x, d_y, d_w, d_sq, rows, cols, kEps);
  });

  CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_ones));
  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));
  CHECK_CUDA(cudaFree(d_w));
  CHECK_CUDA(cudaFree(d_x2));
  CHECK_CUDA(cudaFree(d_sq));
  return ms;
}

static double RunCutlassRMSNorm(const std::vector<float>& h_x,
                                const std::vector<float>& h_weight,
                                std::vector<float>& h_y,
                                int rows, int cols) {
#if HAS_CUTLASS
  // CUTLASS RMSNorm path can be integrated here (custom epilogue / visitor).
  // For now use -1 as explicit placeholder to avoid fake benchmark numbers.
  (void)h_x;
  (void)h_weight;
  (void)h_y;
  (void)rows;
  (void)cols;
  return -1.0;
#else
  (void)h_x;
  (void)h_weight;
  (void)h_y;
  (void)rows;
  (void)cols;
  return -1.0;
#endif
}

template <typename DispatchFn>
static double RunCustomKernel(const std::vector<float>& h_x,
                              const std::vector<float>& h_weight,
                              std::vector<float>& h_y,
                              int rows, int cols,
                              DispatchFn dispatch) {
  int n = rows * cols;
  float *d_x, *d_y, *d_weight;
  CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_y, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_weight, cols * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_weight, h_weight.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

  double ms = MeasureKernelMs([&]() { dispatch(d_x, d_y, d_weight, rows, cols); });
  CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));
  CHECK_CUDA(cudaFree(d_weight));
  return ms;
}

// ============================================================================
// Main
// ============================================================================
int main() {
  std::cout << "============================================================\n";
  std::cout << "  RMSNorm Kernel Performance Comparison (V0/V1/V2/V3/CUB/cuBLAS/CUTLASS)\n";
  std::cout << "============================================================\n\n";

  auto cases = common::LoadOrCreateTestCasesCsv("data/rmsnorm/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/rmsnorm_all_comparison.csv");
  ofs << "id,rows,cols,v0_ms,v1_ms,v2_ms,v3_ms,cub_ms,cublas_ms,cutlass_ms,cudnn_ms,cpu_ms,"
         "v0_check,v1_check,v2_check,v3_check,cub_check,cublas_check\n";

  std::cout << std::left
            << std::setw(8) << "Rows"
            << std::setw(8) << "Cols"
            << std::setw(10) << "V0"
            << std::setw(10) << "V1"
            << std::setw(10) << "V2"
            << std::setw(10) << "V3"
            << std::setw(10) << "CUB"
            << std::setw(10) << "cuBLAS"
            << std::setw(10) << "CUTLASS"
            << std::setw(10) << "cuDNN"
            << std::setw(10) << "CPU"
            << "\n";
  std::cout << std::string(106, '-') << "\n";

  for (size_t i = 0; i < cases.size(); ++i) {
    const int rows = cases[i].rows;
    const int cols = cases[i].cols;
    const int n = rows * cols;

    std::vector<float> h_x(n), h_weight(cols), h_cpu(n), h_out(n);
    common::InitMatrix(h_x, rows, cols);
    common::InitMatrix(h_weight, 1, cols);

    auto t0 = std::chrono::high_resolution_clock::now();
    RMSNormCPU(h_x.data(), h_cpu.data(), h_weight.data(), rows, cols, kEps);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double v0_ms = RunCustomKernel(h_x, h_weight, h_out, rows, cols,
      [&](float* d_x, float* d_y, float* d_w, int r, int c) {
        RMSNormV0Kernel<<<(r + 255) / 256, 256>>>(d_x, d_y, d_w, r, c, kEps);
      });
    bool v0_ok = common::CheckEqual(h_cpu, h_out, 1e-4f);

    double v1_ms = RunCustomKernel(h_x, h_weight, h_out, rows, cols,
      [&](float* d_x, float* d_y, float* d_w, int r, int c) {
        if (CanStageRowV1V2(c)) {
          const std::size_t smem = (static_cast<std::size_t>(c) + kThreadsV1V2) * sizeof(float);
          RMSNormV1StagedKernel<<<r, kThreadsV1V2, smem>>>(d_x, d_y, d_w, r, c, kEps);
        } else {
          RMSNormV1StreamKernel<<<r, kThreadsV1V2>>>(d_x, d_y, d_w, r, c, kEps);
        }
      });
    bool v1_ok = common::CheckEqual(h_cpu, h_out, 1e-4f);

    double v2_ms = RunCustomKernel(h_x, h_weight, h_out, rows, cols,
      [&](float* d_x, float* d_y, float* d_w, int r, int c) {
        if (CanStageRowV1V2(c)) {
          const std::size_t smem = (static_cast<std::size_t>(c) + kThreadsV1V2) * sizeof(float);
          RMSNormV2StagedKernel<<<r, kThreadsV1V2, smem>>>(d_x, d_y, d_w, r, c, kEps);
        } else {
          RMSNormV2StreamKernel<<<r, kThreadsV1V2>>>(d_x, d_y, d_w, r, c, kEps);
        }
      });
    bool v2_ok = common::CheckEqual(h_cpu, h_out, 1e-4f);

    double v3_ms = RunCustomKernel(h_x, h_weight, h_out, rows, cols,
      [&](float* d_x, float* d_y, float* d_w, int r, int c) {
        const int threads = PickThreadsByColsV3(c);
        if (CanStageRowV3(c, threads)) {
          const std::size_t smem = (static_cast<std::size_t>(c) + threads) * sizeof(float);
          RMSNormV3StagedKernel<<<r, threads, smem>>>(d_x, d_y, d_w, r, c, kEps);
        } else {
          RMSNormV3StreamKernel<<<r, threads>>>(d_x, d_y, d_w, r, c, kEps);
        }
      });
    bool v3_ok = common::CheckEqual(h_cpu, h_out, 1e-4f);

    double cub_ms = RunCubRMSNorm(h_x, h_weight, h_out, rows, cols);
    bool cub_ok = common::CheckEqual(h_cpu, h_out, 1e-4f);

    double cublas_ms = RunCuBLASRMSNorm(h_x, h_weight, h_out, rows, cols);
    bool cublas_ok = common::CheckEqual(h_cpu, h_out, 1e-4f);

    double cutlass_ms = RunCutlassRMSNorm(h_x, h_weight, h_out, rows, cols);

    double cudnn_ms = RunCudnnRMSNorm(h_x, h_weight, h_out, rows, cols);

    std::cout << std::fixed << std::setprecision(4)
              << std::setw(8) << rows
              << std::setw(8) << cols
              << std::setw(10) << v0_ms
              << std::setw(10) << v1_ms
              << std::setw(10) << v2_ms
              << std::setw(10) << v3_ms
              << std::setw(10) << cub_ms
              << std::setw(10) << cublas_ms
              << std::setw(10) << cutlass_ms
              << std::setw(10) << cudnn_ms
              << std::setw(10) << cpu_ms
              << "\n";

    ofs << i << "," << rows << "," << cols << ","
        << v0_ms << "," << v1_ms << "," << v2_ms << "," << v3_ms << ","
        << cub_ms << "," << cublas_ms << "," << cutlass_ms << "," << cudnn_ms << "," << cpu_ms << ","
        << (v0_ok ? "PASS" : "FAIL") << ","
        << (v1_ok ? "PASS" : "FAIL") << ","
        << (v2_ok ? "PASS" : "FAIL") << ","
        << (v3_ok ? "PASS" : "FAIL") << ","
        << (cub_ok ? "PASS" : "FAIL") << ","
        << (cublas_ok ? "PASS" : "FAIL")
        << "\n";
  }

  std::cout << "\n============================================================\n";
  std::cout << "Output CSV: data/results/rmsnorm_all_comparison.csv\n";
  std::cout << "============================================================\n";
  return 0;
}
