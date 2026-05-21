#include "gemm/gemm_quant.h"

#include <cuda_fp8.h>

#include "gemm/gemm_api.h"

namespace gemm {
namespace {

#define CKO_QUANT_CUDA(call)                                         \
  do {                                                                \
    const cudaError_t e = (call);                                     \
    if (e != cudaSuccess) {                                          \
      return common::Status::CudaError(cudaGetErrorString(e));        \
    }                                                                 \
  } while (0)

template <typename T>
__device__ T ClampAndConvert(float v) {
  return static_cast<T>(v);
}

// INT8 钳位转换特化
template <>
__device__ int8_t ClampAndConvert<int8_t>(float v) {
  return static_cast<int8_t>(__float2int_rn(fminf(fmaxf(v, -128.f), 127.f)));
}

// FP8E4M3 钳位转换特化
template <>
__device__ __nv_fp8_e4m3 ClampAndConvert<__nv_fp8_e4m3>(float v) {
  return __nv_fp8_e4m3(v);
}

// FP8E5M2 钳位转换特化
template <>
__device__ __nv_fp8_e5m2 ClampAndConvert<__nv_fp8_e5m2>(float v) {
  return __nv_fp8_e5m2(v);
}

// 逐行量化内核：每行独立计算 scale 并量化
template <typename T>
__global__ void QuantizePerRowKernel(const float* __restrict__ in, T* __restrict__ out,
                                     float* __restrict__ scales, int rows, int cols,
                                     float quant_max) {
  constexpr int kWarpSize = 32;
  constexpr int kRowsPerBlock = 4;
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int row = blockIdx.x * kRowsPerBlock + warp_id;

  if (row >= rows) return;

  const float* row_in = in + static_cast<std::size_t>(row) * cols;
  T* row_out = out + static_cast<std::size_t>(row) * cols;

  float max_abs = 0.f;
  for (int c = lane; c < cols; c += kWarpSize) {
    max_abs = fmaxf(max_abs, fabsf(row_in[c]));
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
  }
  max_abs = __shfl_sync(0xffffffff, max_abs, 0);

  float scale = max_abs / quant_max;
  if (scale < 1e-8f) scale = 1.f;
  if (lane == 0) scales[row] = scale;

  for (int c = lane; c < cols; c += kWarpSize) {
    out[static_cast<std::size_t>(row) * cols + c] =
        ClampAndConvert<T>(row_in[c] / scale);
  }
}

// 逐张量最大值查找内核：归约求全局绝对最大值
__global__ void QuantizePerTensorFindMax(const float* __restrict__ in, float* __restrict__ g_max,
                                         int N) {
  __shared__ float smem[256];
  const int tid = threadIdx.x;
  float local = 0.f;
  for (int i = tid + blockIdx.x * blockDim.x; i < N; i += blockDim.x * gridDim.x) {
    local = fmaxf(local, fabsf(in[i]));
  }
  smem[tid] = local;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
    __syncthreads();
  }
  if (tid == 0) atomicMax(reinterpret_cast<int*>(g_max), __float_as_int(smem[0]));
}

// 逐张量量化内核：使用全局 scale 进行量化
template <typename T>
__global__ void QuantizePerTensorQuantize(const float* __restrict__ in, T* __restrict__ out,
                                          int N, float scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  out[idx] = ClampAndConvert<T>(in[idx] / scale);
}

// 逐张量反量化内核（INT32 → FP32）
__global__ void DequantizePerTensorFp32(const int32_t* __restrict__ in, float* __restrict__ out,
                                        int N, float combined_scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  out[idx] = static_cast<float>(in[idx]) * combined_scale;
}

// 逐行 A 反量化内核（INT32 → FP32）：每行使用不同的 scale_a
__global__ void DequantizePerRowAFp32(const int32_t* __restrict__ in, float* __restrict__ out,
                                      int M, int N, const float* __restrict__ scale_a,
                                      float scale_b) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  const int idx = static_cast<std::size_t>(row) * N + col;
  out[idx] = static_cast<float>(in[idx]) * scale_a[row] * scale_b;
}

// 逐张量反量化内核（浮点 → FP32）
__global__ void DequantizePerTensorFloat(const float* __restrict__ in, float* __restrict__ out,
                                         int N, float combined_scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  out[idx] = in[idx] * combined_scale;
}

// 逐行 A 反量化内核（浮点 → FP32）：每行使用不同的 scale_a
__global__ void DequantizePerRowAFloat(const float* __restrict__ in, float* __restrict__ out,
                                       int M, int N, const float* __restrict__ scale_a,
                                       float scale_b) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  const int idx = static_cast<std::size_t>(row) * N + col;
  out[idx] = in[idx] * scale_a[row] * scale_b;
}

}  // namespace

void LaunchQuantizeMatrix(const float* d_in, void* d_out, float* d_scale,
                          int rows, int cols, common::DType dtype,
                          QuantScheme scheme, cudaStream_t stream) {
  const int threads = 256;
  const int n_elem = rows * cols;
  const float qmax = QuantMax(dtype);

  // PerRow 量化路径
  if (scheme == QuantScheme::kPerRow) {
    const int rows_per_block = 4;
    const int grid = (rows + rows_per_block - 1) / rows_per_block;
    switch (dtype) {
      case common::DType::kInt8:
        QuantizePerRowKernel<<<grid, threads, 0, stream>>>(
            d_in, reinterpret_cast<int8_t*>(d_out), d_scale, rows, cols, qmax); break;
      case common::DType::kFp8E4M3:
        QuantizePerRowKernel<<<grid, threads, 0, stream>>>(
            d_in, reinterpret_cast<__nv_fp8_e4m3*>(d_out), d_scale, rows, cols, qmax); break;
      case common::DType::kFp8E5M2:
        QuantizePerRowKernel<<<grid, threads, 0, stream>>>(
            d_in, reinterpret_cast<__nv_fp8_e5m2*>(d_out), d_scale, rows, cols, qmax); break;
      default: break;
    }
    return;
  }

  // PerTensor 量化路径
  const int max_blocks = 512;
  const int red_grid = std::min(max_blocks, (n_elem + threads - 1) / threads);

  float h_max;
  float* d_max;
  cudaMalloc(&d_max, sizeof(float));
  cudaMemset(d_max, 0, sizeof(float));

  QuantizePerTensorFindMax<<<red_grid, threads, 0, stream>>>(d_in, d_max, n_elem);

  cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
  float scale = h_max / qmax;
  if (scale < 1e-8f) scale = 1.f;

  cudaMemcpy(d_scale, &scale, sizeof(float), cudaMemcpyHostToDevice);

  const int q_grid = (n_elem + threads - 1) / threads;
  switch (dtype) {
    case common::DType::kInt8:
      QuantizePerTensorQuantize<<<q_grid, threads, 0, stream>>>(
          d_in, reinterpret_cast<int8_t*>(d_out), n_elem, scale); break;
    case common::DType::kFp8E4M3:
      QuantizePerTensorQuantize<<<q_grid, threads, 0, stream>>>(
          d_in, reinterpret_cast<__nv_fp8_e4m3*>(d_out), n_elem, scale); break;
    case common::DType::kFp8E5M2:
      QuantizePerTensorQuantize<<<q_grid, threads, 0, stream>>>(
          d_in, reinterpret_cast<__nv_fp8_e5m2*>(d_out), n_elem, scale); break;
    default: break;
  }

  cudaFree(d_max);
}

void LaunchDequantizeGemmOutput(const void* d_in, float* d_out,
                                const float* d_scale_a, const float* d_scale_b,
                                int M, int N, common::DType compute_dtype,
                                QuantScheme scheme_a, QuantScheme scheme_b,
                                cudaStream_t stream) {
  const int threads = 256;
  const int n_elem = M * N;

  const bool is_int8 = (compute_dtype == common::DType::kInt8);
  const bool per_tensor = (scheme_a == QuantScheme::kPerTensor &&
                           scheme_b == QuantScheme::kPerTensor);

  if (per_tensor) {
    float h_scale_a, h_scale_b;
    cudaMemcpy(&h_scale_a, d_scale_a, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_scale_b, d_scale_b, sizeof(float), cudaMemcpyDeviceToHost);
    const float combined = h_scale_a * h_scale_b;
    const int grid = (n_elem + threads - 1) / threads;
    if (is_int8) {
      DequantizePerTensorFp32<<<grid, threads, 0, stream>>>(
          reinterpret_cast<const int32_t*>(d_in), d_out, n_elem, combined);
    } else {
      DequantizePerTensorFloat<<<grid, threads, 0, stream>>>(
          reinterpret_cast<const float*>(d_in), d_out, n_elem, combined);
    }
  } else {
    float h_scale_b;
    cudaMemcpy(&h_scale_b, d_scale_b, sizeof(float), cudaMemcpyDeviceToHost);
    const dim3 block(threads);
    const dim3 grid((N + threads - 1) / threads, M);
    if (is_int8) {
      DequantizePerRowAFp32<<<grid, block, 0, stream>>>(
          reinterpret_cast<const int32_t*>(d_in), d_out, M, N, d_scale_a, h_scale_b);
    } else {
      DequantizePerRowAFloat<<<grid, block, 0, stream>>>(
          reinterpret_cast<const float*>(d_in), d_out, M, N, d_scale_a, h_scale_b);
    }
  }
}

common::Status GemmQuantizedRun(const float* d_A_fp32, const float* d_B_fp32,
                                float* d_C_fp32, int M, int N, int K,
                                common::DType quant_dtype, QuantScheme scheme_a,
                                QuantScheme scheme_b, cudaStream_t stream) {
  // 对齐校验：未对齐则回退到 FP32 GEMM
  const bool is_int8 = (quant_dtype == common::DType::kInt8);
  const int align_req = is_int8 ? 16 : 8;

  if ((M % align_req != 0) || (N % align_req != 0) || (K % align_req != 0)) {
    gemm::GemmParams p;
    p.M = M; p.N = N; p.K = K;
    p.dtype_a = common::DType::kFp32;
    p.dtype_b = common::DType::kFp32;
    p.dtype_c = common::DType::kFp32;
    p.layout = common::Layout::kRowMajor;
    p.A = const_cast<float*>(d_A_fp32);
    p.B = const_cast<float*>(d_B_fp32);
    p.C = d_C_fp32;
    p.lda = K; p.ldb = N; p.ldc = N;
    p.impl = ImplId::kAuto;
    return GemmRun(p, stream);
  }

  // 创建或复用 CUDA 流
  cudaStream_t s = stream;
  if (!s) {
    CKO_QUANT_CUDA(cudaStreamCreate(&s));
  }

  // 分配量化缓冲区
  void* d_A_quant = nullptr;
  void* d_B_quant = nullptr;
  void* d_C_int = nullptr;
  float* d_scale_a = nullptr;
  float* d_scale_b = nullptr;

  const size_t elem_a = static_cast<size_t>(M) * K;
  const size_t elem_b = static_cast<size_t>(K) * N;
  const size_t elem_c = static_cast<size_t>(M) * N;
  const int elem_bytes = 1;
  const size_t c_elem_bytes = is_int8 ? sizeof(int32_t) : sizeof(float);

  CKO_QUANT_CUDA(cudaMalloc(&d_A_quant, elem_a * elem_bytes));
  CKO_QUANT_CUDA(cudaMalloc(&d_B_quant, elem_b * elem_bytes));
  CKO_QUANT_CUDA(cudaMalloc(&d_C_int, elem_c * c_elem_bytes));
  const int n_scale_a = (scheme_a == QuantScheme::kPerRow) ? M : 1;
  const int n_scale_b = (scheme_b == QuantScheme::kPerRow) ? K : 1;
  CKO_QUANT_CUDA(cudaMalloc(&d_scale_a, n_scale_a * sizeof(float)));
  CKO_QUANT_CUDA(cudaMalloc(&d_scale_b, n_scale_b * sizeof(float)));

  // 执行量化
  LaunchQuantizeMatrix(d_A_fp32, d_A_quant, d_scale_a, M, K, quant_dtype, scheme_a, s);
  LaunchQuantizeMatrix(d_B_fp32, d_B_quant, d_scale_b, K, N, quant_dtype, scheme_b, s);

  // 构造 GEMM 参数并执行
  gemm::GemmParams p;
  p.M = M; p.N = N; p.K = K;
  p.dtype_a = quant_dtype;
  p.dtype_b = quant_dtype;
  p.dtype_c = is_int8 ? common::DType::kInt32 : common::DType::kFp32;
  p.layout = common::Layout::kRowMajor;
  p.A = d_A_quant; p.B = d_B_quant; p.C = d_C_int;
  p.lda = K; p.ldb = N; p.ldc = N;
  p.impl = ImplId::kAuto;

  common::Status st = GemmRun(p, s);
  if (!st.ok()) {
    cudaFree(d_A_quant); cudaFree(d_B_quant); cudaFree(d_C_int);
    cudaFree(d_scale_a); cudaFree(d_scale_b);
    if (!stream) cudaStreamDestroy(s);
    return st;
  }

  // 反量化输出
  LaunchDequantizeGemmOutput(d_C_int, d_C_fp32, d_scale_a, d_scale_b,
                             M, N, quant_dtype, scheme_a, scheme_b, s);

  cudaFree(d_A_quant); cudaFree(d_B_quant); cudaFree(d_C_int);
  cudaFree(d_scale_a); cudaFree(d_scale_b);

  // 清理临时流
  if (!stream) {
    CKO_QUANT_CUDA(cudaStreamSynchronize(s));
    CKO_QUANT_CUDA(cudaStreamDestroy(s));
  }

  return common::Status::Ok();
}

}  // namespace gemm
