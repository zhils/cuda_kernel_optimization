#include "rmsnorm/rmsnorm_gpu.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <string>

#include "rmsnorm/rmsnorm_dtype.h"
#include "rmsnorm/rmsnorm_quant.h"
#include "rmsnorm/rmsnorm_v3_dtype.cuh"
#include "rmsnorm_kernels.cuh"

namespace rmsnorm {
namespace {

#define CKO_CUDA(call)                                                         \
  do {                                                                         \
    const cudaError_t err__ = (call);                                          \
    if (err__ != cudaSuccess) {                                                \
      return common::Status::CudaError(std::string(cudaGetErrorString(err__))); \
    }                                                                          \
  } while (0)

ActivationDtype ToActivationDtype(common::DType dt) {
  switch (dt) {
    case common::DType::kFp32: return ActivationDtype::kFp32;
    case common::DType::kFp16: return ActivationDtype::kFp16;
    case common::DType::kBf16: return ActivationDtype::kBf16;
    case common::DType::kFp8E4M3: return ActivationDtype::kFp8E4M3;
    case common::DType::kFp8E5M2: return ActivationDtype::kFp8E5M2;
    case common::DType::kInt8: return ActivationDtype::kInt8;
  }
  return ActivationDtype::kFp32;
}

WeightDtype ToWeightDtype(common::DType dt) {
  switch (dt) {
    case common::DType::kFp32: return WeightDtype::kFp32;
    case common::DType::kFp16: return WeightDtype::kFp16;
    case common::DType::kBf16: return WeightDtype::kBf16;
    default: return WeightDtype::kFp32;
  }
}

ImplId ResolveImpl(const RmsNormParams& p) {
  if (p.impl != ImplId::kAuto) return p.impl;
  return ImplId::kV3;
}

bool IsLegacyFp32Only(const RmsNormParams& p) {
  return p.act_dtype == common::DType::kFp32 && p.weight_dtype == common::DType::kFp32;
}

constexpr int kCubBlockSize = 256;

// CUB 归约参考实现内核
__global__ void RMSNormCUBKernel(const float* __restrict__ x, float* __restrict__ y,
                                 const float* __restrict__ weight, int rows, int cols,
                                 float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float* row_x = x + static_cast<size_t>(row) * cols;
  float* row_y = y + static_cast<size_t>(row) * cols;

  float local_sum = 0.f;
  const int cols4 = cols / 4;
  const bool align4 = (cols % 4 == 0) &&
                      (reinterpret_cast<uintptr_t>(row_x) % 16u == 0u) &&
                      (reinterpret_cast<uintptr_t>(row_y) % 16u == 0u);
  if (align4) {
    for (int c = threadIdx.x; c < cols4; c += blockDim.x) {
      const float4 v = reinterpret_cast<const float4*>(row_x)[c];
      local_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
  } else {
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
      const float val = row_x[c];
      local_sum += val * val;
    }
  }

  typedef cub::BlockReduce<float, kCubBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp;
  const float sq_sum = BlockReduce(temp).Sum(local_sum);

  __shared__ float s_rms;
  if (threadIdx.x == 0) {
    s_rms = rsqrtf(sq_sum / static_cast<float>(cols) + eps);
  }
  __syncthreads();
  const float rms = s_rms;

  if (align4) {
    for (int c = threadIdx.x; c < cols4; c += blockDim.x) {
      const float4 vx = reinterpret_cast<const float4*>(row_x)[c];
      const float4 vw = reinterpret_cast<const float4*>(weight)[c];
      reinterpret_cast<float4*>(row_y)[c] =
          make_float4(vx.x * rms * vw.x, vx.y * rms * vw.y, vx.z * rms * vw.z,
                      vx.w * rms * vw.w);
    }
  } else {
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
      row_y[c] = row_x[c] * rms * weight[c];
    }
  }
}

common::Status LaunchV0(const RmsNormParams& p, cudaStream_t stream) {
  const auto* x = reinterpret_cast<const float*>(p.input);
  auto* y = reinterpret_cast<float*>(p.output);
  const auto* weight = reinterpret_cast<const float*>(p.weight);
  const dim3 grid((p.rows + 255) / 256);
  const dim3 block(256);
  RMSNormV0Kernel<<<grid, block, 0, stream>>>(x, y, weight, p.rows, p.cols, p.eps);
  return common::Status::Ok();
}

common::Status LaunchV1(const RmsNormParams& p, cudaStream_t stream) {
  const auto* x = reinterpret_cast<const float*>(p.input);
  auto* y = reinterpret_cast<float*>(p.output);
  const auto* weight = reinterpret_cast<const float*>(p.weight);
  const size_t smem = static_cast<size_t>(p.cols) * sizeof(float) * RMSNORM_WARPS_PER_BLOCK;
  CKO_CUDA(cudaFuncSetAttribute(RMSNormV1Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(smem)));
  const dim3 block(RMSNORM_BLOCK_SIZE);
  const dim3 grid((p.rows + RMSNORM_WARPS_PER_BLOCK - 1) / RMSNORM_WARPS_PER_BLOCK);
  RMSNormV1Kernel<<<grid, block, smem, stream>>>(x, y, weight, p.rows, p.cols, p.eps);
  return common::Status::Ok();
}

common::Status LaunchV2(const RmsNormParams& p, cudaStream_t stream) {
  const auto* x = reinterpret_cast<const float*>(p.input);
  auto* y = reinterpret_cast<float*>(p.output);
  const auto* weight = reinterpret_cast<const float*>(p.weight);
  const size_t smem = static_cast<size_t>(p.cols) * sizeof(float) * RMSNORM_WARPS_PER_BLOCK;
  CKO_CUDA(cudaFuncSetAttribute(RMSNormV2Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(smem)));
  const dim3 block(RMSNORM_BLOCK_SIZE);
  const dim3 grid((p.rows + RMSNORM_WARPS_PER_BLOCK - 1) / RMSNORM_WARPS_PER_BLOCK);
  RMSNormV2Kernel<<<grid, block, smem, stream>>>(x, y, weight, p.rows, p.cols, p.eps);
  return common::Status::Ok();
}

common::Status LaunchCubRef(const RmsNormParams& p, cudaStream_t stream) {
  const auto* x = reinterpret_cast<const float*>(p.input);
  auto* y = reinterpret_cast<float*>(p.output);
  const auto* weight = reinterpret_cast<const float*>(p.weight);
  const dim3 grid(p.rows);
  const dim3 block(kCubBlockSize);
  RMSNormCUBKernel<<<grid, block, 0, stream>>>(x, y, weight, p.rows, p.cols, p.eps);
  return common::Status::Ok();
}

common::Status LaunchV3(const RmsNormParams& p, cudaStream_t stream) {
  const ActivationDtype act = ToActivationDtype(p.act_dtype);
  const WeightDtype wdt = ToWeightDtype(p.weight_dtype);
  const size_t smem = static_cast<size_t>(p.cols) * sizeof(float);
  const dim3 block(kV3BlockSize);
  const dim3 grid((p.rows + kV3WarpsPerBlock - 1) / kV3WarpsPerBlock);

  // FP32 激活路径
  if (act == ActivationDtype::kFp32) {
    CKO_CUDA(cudaFuncSetAttribute(RMSNormV3KernelFp32,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(smem)));
    RMSNormV3KernelFp32<<<grid, block, smem, stream>>>(
        reinterpret_cast<const float*>(p.input), reinterpret_cast<float*>(p.output),
        const_cast<void*>(p.weight), wdt, p.rows, p.cols, p.eps);
    return common::Status::Ok();
  }

  // FP16 激活路径
  if (act == ActivationDtype::kFp16) {
    CKO_CUDA(cudaFuncSetAttribute(RMSNormV3KernelFp16,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(smem)));
    RMSNormV3KernelFp16<<<grid, block, smem, stream>>>(
        reinterpret_cast<const __half*>(p.input), reinterpret_cast<__half*>(p.output),
        const_cast<void*>(p.weight), wdt, p.rows, p.cols, p.eps);
    return common::Status::Ok();
  }

  // BF16 激活路径
  if (act == ActivationDtype::kBf16) {
    CKO_CUDA(cudaFuncSetAttribute(RMSNormV3KernelBf16,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(smem)));
    RMSNormV3KernelBf16<<<grid, block, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(p.input),
        reinterpret_cast<__nv_bfloat16*>(p.output), const_cast<void*>(p.weight), wdt, p.rows,
        p.cols, p.eps);
    return common::Status::Ok();
  }

  // 量化激活路径：INT8
  const float quant_max = QuantMax(act);
  if (act == ActivationDtype::kInt8) {
    CKO_CUDA(cudaFuncSetAttribute(RMSNormV3KernelInt8,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(smem)));
    RMSNormV3KernelInt8<<<grid, block, smem, stream>>>(
        reinterpret_cast<const int8_t*>(p.input), reinterpret_cast<int8_t*>(p.output),
        p.input_scale, p.output_scale, const_cast<void*>(p.weight), wdt, p.rows, p.cols, p.eps,
        quant_max);
    return common::Status::Ok();
  }

  // 量化激活路径：FP8E4M3
  if (act == ActivationDtype::kFp8E4M3) {
    CKO_CUDA(cudaFuncSetAttribute(RMSNormV3KernelFp8E4M3,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(smem)));
    RMSNormV3KernelFp8E4M3<<<grid, block, smem, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(p.input),
        reinterpret_cast<__nv_fp8_e4m3*>(p.output), p.input_scale, p.output_scale,
        const_cast<void*>(p.weight), wdt, p.rows, p.cols, p.eps, quant_max);
    return common::Status::Ok();
  }

  // 量化激活路径：FP8E5M2
  if (act == ActivationDtype::kFp8E5M2) {
    CKO_CUDA(cudaFuncSetAttribute(RMSNormV3KernelFp8E5M2,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(smem)));
    RMSNormV3KernelFp8E5M2<<<grid, block, smem, stream>>>(
        reinterpret_cast<const __nv_fp8_e5m2*>(p.input),
        reinterpret_cast<__nv_fp8_e5m2*>(p.output), p.input_scale, p.output_scale,
        const_cast<void*>(p.weight), wdt, p.rows, p.cols, p.eps, quant_max);
    return common::Status::Ok();
  }

  return common::Status::Unsupported("activation dtype not supported");
}

#undef CKO_CUDA

}  // namespace

common::Status RmsNormRunGpu(const RmsNormParams& p, cudaStream_t stream) {
  const ImplId impl = ResolveImpl(p);

  // FP32 专用实现（V0 / V1 / V2 / CUB）
  switch (impl) {
    case ImplId::kV0:
      if (!IsLegacyFp32Only(p)) {
        return common::Status::Unsupported("v0 supports fp32 activation and fp32 weight only");
      }
      return LaunchV0(p, stream);
    case ImplId::kV1:
      if (!IsLegacyFp32Only(p)) {
        return common::Status::Unsupported("v1 supports fp32 activation and fp32 weight only");
      }
      return LaunchV1(p, stream);
    case ImplId::kV2:
      if (!IsLegacyFp32Only(p)) {
        return common::Status::Unsupported("v2 supports fp32 activation and fp32 weight only");
      }
      return LaunchV2(p, stream);
    case ImplId::kCubRef:
      if (!IsLegacyFp32Only(p)) {
        return common::Status::Unsupported("cub_ref supports fp32 activation and fp32 weight only");
      }
      return LaunchCubRef(p, stream);

    // 多类型通用实现（V3 / Auto）
    case ImplId::kV3:
    case ImplId::kAuto:
      return LaunchV3(p, stream);
  }
  return common::Status::InvalidArgument("unknown impl id");
}

}  // namespace rmsnorm
