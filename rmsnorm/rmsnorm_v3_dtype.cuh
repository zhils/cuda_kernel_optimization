#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstddef>
#include <cstdint>

#include "rmsnorm/rmsnorm_dtype.h"

namespace rmsnorm {

constexpr int kV3WarpSize = 32;
constexpr int kV3BlockSize = 128;
constexpr int kV3WarpsPerBlock = 4;

// Warp 级归约函数
__device__ __forceinline__ float WarpReduceSum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return __shfl_sync(0xffffffff, val, 0);
}

__device__ __forceinline__ float WarpReduceMax(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return __shfl_sync(0xffffffff, val, 0);
}

// 设备端量化与反量化函数
__device__ __forceinline__ int8_t QuantizeInt8Device(float v, float scale) {
  const float q = v / scale;
  const float clamped = fminf(fmaxf(q, -128.f), 127.f);
  return static_cast<int8_t>(__float2int_rn(clamped));
}

__device__ __forceinline__ float DequantizeInt8Device(int8_t q, float scale) {
  return static_cast<float>(q) * scale;
}

__device__ __forceinline__ __nv_fp8_e4m3 QuantizeFp8E4M3Device(float v, float scale) {
  return __nv_fp8_e4m3(v / scale);
}

__device__ __forceinline__ float DequantizeFp8E4M3Device(__nv_fp8_e4m3 q, float scale) {
  return static_cast<float>(q) * scale;
}

__device__ __forceinline__ __nv_fp8_e5m2 QuantizeFp8E5M2Device(float v, float scale) {
  return __nv_fp8_e5m2(v / scale);
}

__device__ __forceinline__ float DequantizeFp8E5M2Device(__nv_fp8_e5m2 q, float scale) {
  return static_cast<float>(q) * scale;
}

// V3 内核：fp32 激活，weight 类型由 weight_dtype 参数决定
__global__ void RMSNormV3KernelFp32(const float* __restrict__ x,
                                    float* __restrict__ y,
                                    const void* __restrict__ weight,
                                    WeightDtype weight_dtype,
                                    int rows, int cols, float eps) {
  const int warp_id = threadIdx.x / kV3WarpSize;
  const int lane = threadIdx.x % kV3WarpSize;
  const int row = blockIdx.x * kV3WarpsPerBlock + warp_id;

  extern __shared__ float s_weight[];
  for (int c = threadIdx.x; c < cols; c += kV3BlockSize) {
    s_weight[c] = LoadWeightToFloat(weight, weight_dtype, c);
  }
  __syncthreads();

  if (row >= rows) return;

  const float* row_x = x + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  float* row_y = y + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);

  const int cols4 = cols / 4;
  const bool align4 = (cols % 4 == 0) &&
                      (reinterpret_cast<std::uintptr_t>(row_x) % 16u == 0u) &&
                      (reinterpret_cast<std::uintptr_t>(row_y) % 16u == 0u);

  float local_sum = 0.f;
  if (align4) {
    for (int c = lane; c < cols4; c += kV3WarpSize) {
      const float4 v = __ldg(reinterpret_cast<const float4*>(row_x + c * 4));
      local_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
  } else {
    for (int c = lane; c < cols; c += kV3WarpSize) {
      const float val = __ldg(row_x + c);
      local_sum += val * val;
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  }
  const float sq_sum = __shfl_sync(0xffffffff, local_sum, 0);
  const float rms = rsqrtf(sq_sum / static_cast<float>(cols) + eps);

  if (align4) {
    for (int c = lane; c < cols4; c += kV3WarpSize) {
      const float4 vx = __ldg(reinterpret_cast<const float4*>(row_x + c * 4));
      const float4 vw = *reinterpret_cast<const float4*>(s_weight + c * 4);
      *reinterpret_cast<float4*>(row_y + c * 4) =
          make_float4(vx.x * rms * vw.x, vx.y * rms * vw.y, vx.z * rms * vw.z,
                      vx.w * rms * vw.w);
    }
  } else {
    for (int c = lane; c < cols; c += kV3WarpSize) {
      row_y[c] = __ldg(row_x + c) * rms * s_weight[c];
    }
  }
}

// V3 内核：fp16 激活，weight 类型由 weight_dtype 参数决定
__global__ void RMSNormV3KernelFp16(const __half* __restrict__ x,
                                    __half* __restrict__ y,
                                    const void* __restrict__ weight,
                                    WeightDtype weight_dtype,
                                    int rows, int cols, float eps) {
  const int warp_id = threadIdx.x / kV3WarpSize;
  const int lane = threadIdx.x % kV3WarpSize;
  const int row = blockIdx.x * kV3WarpsPerBlock + warp_id;

  extern __shared__ float s_weight[];
  for (int c = threadIdx.x; c < cols; c += kV3BlockSize) {
    s_weight[c] = LoadWeightToFloat(weight, weight_dtype, c);
  }
  __syncthreads();

  if (row >= rows) return;

  const __half* row_x = x + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  __half* row_y = y + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);

  float local_sum = 0.f;
  for (int c = lane; c < cols; c += kV3WarpSize) {
    const float val = __half2float(__ldg(row_x + c));
    local_sum += val * val;
  }
  const float rms = rsqrtf(WarpReduceSum(local_sum) / static_cast<float>(cols) + eps);

  for (int c = lane; c < cols; c += kV3WarpSize) {
    const float val = __half2float(__ldg(row_x + c));
    row_y[c] = __float2half(val * rms * s_weight[c]);
  }
}

// V3 内核：bf16 激活，weight 类型由 weight_dtype 参数决定
__global__ void RMSNormV3KernelBf16(const __nv_bfloat16* __restrict__ x,
                                    __nv_bfloat16* __restrict__ y,
                                    const void* __restrict__ weight,
                                    WeightDtype weight_dtype,
                                    int rows, int cols, float eps) {
  const int warp_id = threadIdx.x / kV3WarpSize;
  const int lane = threadIdx.x % kV3WarpSize;
  const int row = blockIdx.x * kV3WarpsPerBlock + warp_id;

  extern __shared__ float s_weight[];
  for (int c = threadIdx.x; c < cols; c += kV3BlockSize) {
    s_weight[c] = LoadWeightToFloat(weight, weight_dtype, c);
  }
  __syncthreads();

  if (row >= rows) return;

  const __nv_bfloat16* row_x =
      x + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  __nv_bfloat16* row_y = y + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);

  float local_sum = 0.f;
  for (int c = lane; c < cols; c += kV3WarpSize) {
    const float val = __bfloat162float(__ldg(row_x + c));
    local_sum += val * val;
  }
  const float rms = rsqrtf(WarpReduceSum(local_sum) / static_cast<float>(cols) + eps);

  for (int c = lane; c < cols; c += kV3WarpSize) {
    const float val = __bfloat162float(__ldg(row_x + c));
    row_y[c] = __float2bfloat16(val * rms * s_weight[c]);
  }
}

// V3 内核：int8 量化激活，含反量化、归约、动态量化输出
__global__ void RMSNormV3KernelInt8(const int8_t* __restrict__ x,
                                    int8_t* __restrict__ y,
                                    const float* __restrict__ x_scale,
                                    float* __restrict__ y_scale,
                                    const void* __restrict__ weight,
                                    WeightDtype weight_dtype,
                                    int rows, int cols, float eps, float quant_max) {
  const int warp_id = threadIdx.x / kV3WarpSize;
  const int lane = threadIdx.x % kV3WarpSize;
  const int row = blockIdx.x * kV3WarpsPerBlock + warp_id;

  extern __shared__ float s_weight[];
  for (int c = threadIdx.x; c < cols; c += kV3BlockSize) {
    s_weight[c] = LoadWeightToFloat(weight, weight_dtype, c);
  }
  __syncthreads();

  if (row >= rows) return;

  const int8_t* row_x = x + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  int8_t* row_y = y + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  const float in_scale = x_scale[row];

  float local_sum = 0.f;
  for (int c = lane; c < cols; c += kV3WarpSize) {
    const float val = DequantizeInt8Device(row_x[c], in_scale);
    local_sum += val * val;
  }
  const float rms = rsqrtf(WarpReduceSum(local_sum) / static_cast<float>(cols) + eps);

  float local_max = 0.f;
  for (int c = lane; c < cols; c += kV3WarpSize) {
    const float val = DequantizeInt8Device(row_x[c], in_scale);
    const float yf = val * rms * s_weight[c];
    local_max = fmaxf(local_max, fabsf(yf));
  }
  const float row_max = WarpReduceMax(local_max);
  float out_scale = row_max / quant_max;
  if (out_scale < 1e-8f) out_scale = 1.f;
  if (lane == 0) y_scale[row] = out_scale;

  for (int c = lane; c < cols; c += kV3WarpSize) {
    const float val = DequantizeInt8Device(row_x[c], in_scale);
    const float yf = val * rms * s_weight[c];
    row_y[c] = QuantizeInt8Device(yf, out_scale);
  }
}

// V3 内核：fp8_e4m3 量化激活，含反量化、归约、动态量化输出
__global__ void RMSNormV3KernelFp8E4M3(const __nv_fp8_e4m3* __restrict__ x,
                                       __nv_fp8_e4m3* __restrict__ y,
                                       const float* __restrict__ x_scale,
                                       float* __restrict__ y_scale,
                                       const void* __restrict__ weight,
                                       WeightDtype weight_dtype,
                                       int rows, int cols, float eps, float quant_max) {
  const int warp_id = threadIdx.x / kV3WarpSize;
  const int lane = threadIdx.x % kV3WarpSize;
  const int row = blockIdx.x * kV3WarpsPerBlock + warp_id;

  extern __shared__ float s_weight[];
  for (int c = threadIdx.x; c < cols; c += kV3BlockSize) {
    s_weight[c] = LoadWeightToFloat(weight, weight_dtype, c);
  }
  __syncthreads();

  if (row >= rows) return;

  const __nv_fp8_e4m3* row_x =
      x + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  __nv_fp8_e4m3* row_y = y + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  const float in_scale = x_scale[row];

  float local_sum = 0.f;
  for (int c = lane; c < cols; c += kV3WarpSize) {
    const float val = DequantizeFp8E4M3Device(row_x[c], in_scale);
    local_sum += val * val;
  }
  const float rms = rsqrtf(WarpReduceSum(local_sum) / static_cast<float>(cols) + eps);

  float local_max = 0.f;
  for (int c = lane; c < cols; c += kV3WarpSize) {
    const float val = DequantizeFp8E4M3Device(row_x[c], in_scale);
    const float yf = val * rms * s_weight[c];
    local_max = fmaxf(local_max, fabsf(yf));
  }
  const float row_max = WarpReduceMax(local_max);
  float out_scale = row_max / quant_max;
  if (out_scale < 1e-8f) out_scale = 1.f;
  if (lane == 0) y_scale[row] = out_scale;

  for (int c = lane; c < cols; c += kV3WarpSize) {
    const float val = DequantizeFp8E4M3Device(row_x[c], in_scale);
    const float yf = val * rms * s_weight[c];
    row_y[c] = QuantizeFp8E4M3Device(yf, out_scale);
  }
}

// V3 内核：fp8_e5m2 量化激活，含反量化、归约、动态量化输出
__global__ void RMSNormV3KernelFp8E5M2(const __nv_fp8_e5m2* __restrict__ x,
                                       __nv_fp8_e5m2* __restrict__ y,
                                       const float* __restrict__ x_scale,
                                       float* __restrict__ y_scale,
                                       const void* __restrict__ weight,
                                       WeightDtype weight_dtype,
                                       int rows, int cols, float eps, float quant_max) {
  const int warp_id = threadIdx.x / kV3WarpSize;
  const int lane = threadIdx.x % kV3WarpSize;
  const int row = blockIdx.x * kV3WarpsPerBlock + warp_id;

  extern __shared__ float s_weight[];
  for (int c = threadIdx.x; c < cols; c += kV3BlockSize) {
    s_weight[c] = LoadWeightToFloat(weight, weight_dtype, c);
  }
  __syncthreads();

  if (row >= rows) return;

  const __nv_fp8_e5m2* row_x =
      x + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  __nv_fp8_e5m2* row_y = y + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  const float in_scale = x_scale[row];

  float local_sum = 0.f;
  for (int c = lane; c < cols; c += kV3WarpSize) {
    const float val = DequantizeFp8E5M2Device(row_x[c], in_scale);
    local_sum += val * val;
  }
  const float rms = rsqrtf(WarpReduceSum(local_sum) / static_cast<float>(cols) + eps);

  float local_max = 0.f;
  for (int c = lane; c < cols; c += kV3WarpSize) {
    const float val = DequantizeFp8E5M2Device(row_x[c], in_scale);
    const float yf = val * rms * s_weight[c];
    local_max = fmaxf(local_max, fabsf(yf));
  }
  const float row_max = WarpReduceMax(local_max);
  float out_scale = row_max / quant_max;
  if (out_scale < 1e-8f) out_scale = 1.f;
  if (lane == 0) y_scale[row] = out_scale;

  for (int c = lane; c < cols; c += kV3WarpSize) {
    const float val = DequantizeFp8E5M2Device(row_x[c], in_scale);
    const float yf = val * rms * s_weight[c];
    row_y[c] = QuantizeFp8E5M2Device(yf, out_scale);
  }
}

}  // namespace rmsnorm
