#include "gemm/gemm_gpu.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <mma.h>

#include <cstdint>
#include <string>

#include "gemm_fp16_kernel.cuh"
#include "gemm_kernels.cuh"
#include "gemm_v3_kernel.cuh"

#if defined(CKO_GEMM_CUTLASS_FP16)
#include "cutlass/cutlass.h"
#include "gemm_cute_fp16.cuh"
#endif

namespace gemm {
namespace {

#define CKO_CUDA(call)                                                         \
  do {                                                                         \
    const cudaError_t err__ = (call);                                          \
    if (err__ != cudaSuccess) {                                                \
      return common::Status::CudaError(std::string(cudaGetErrorString(err__))); \
    }                                                                          \
  } while (0)

bool IsFp16Gemm(const GemmParams& p) {
  return p.dtype_a == common::DType::kFp16 && p.dtype_b == common::DType::kFp16 &&
         p.dtype_c == common::DType::kFp32;
}

bool IsBf16Gemm(const GemmParams& p) {
  return p.dtype_a == common::DType::kBf16 && p.dtype_b == common::DType::kBf16 &&
         p.dtype_c == common::DType::kFp32;
}

bool IsInt8Gemm(const GemmParams& p) {
  return p.dtype_a == common::DType::kInt8 && p.dtype_b == common::DType::kInt8 &&
         (p.dtype_c == common::DType::kFp32 || p.dtype_c == common::DType::kInt32);
}

bool IsFp8Gemm(const GemmParams& p) {
  return (p.dtype_a == common::DType::kFp8E4M3 || p.dtype_a == common::DType::kFp8E5M2) &&
         p.dtype_a == p.dtype_b && p.dtype_c == common::DType::kFp32;
}

namespace {

__global__ void Int8ToFp32Kernel(const int8_t* in, float* out, int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) out[idx] = static_cast<float>(in[idx]);
}
__global__ void Fp16ToFp32Kernel(const __half* in, float* out, int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) out[idx] = __half2float(in[idx]);
}
__global__ void Bf16ToFp32Kernel(const __nv_bfloat16* in, float* out, int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) out[idx] = __bfloat162float(in[idx]);
}
__global__ void Fp8E4M3ToFp32Kernel(const __nv_fp8_e4m3* in, float* out, int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) out[idx] = static_cast<float>(in[idx]);
}
__global__ void Fp8E5M2ToFp32Kernel(const __nv_fp8_e5m2* in, float* out, int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) out[idx] = static_cast<float>(in[idx]);
}

void LaunchConvertToFp32(const void* d_in, float* d_out, int N, common::DType dtype,
                         cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  switch (dtype) {
    case common::DType::kInt8:
      Int8ToFp32Kernel<<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const int8_t*>(d_in), d_out, N); break;
    case common::DType::kFp16:
      Fp16ToFp32Kernel<<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const __half*>(d_in), d_out, N); break;
    case common::DType::kBf16:
      Bf16ToFp32Kernel<<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(d_in), d_out, N); break;
    case common::DType::kFp8E4M3:
      Fp8E4M3ToFp32Kernel<<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const __nv_fp8_e4m3*>(d_in), d_out, N); break;
    case common::DType::kFp8E5M2:
      Fp8E5M2ToFp32Kernel<<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const __nv_fp8_e5m2*>(d_in), d_out, N); break;
    default: break;
  }
}

}  // namespace

common::Status RunV0(const GemmParams& p, cudaStream_t stream) {
  // 参数校验
  if (IsFp16Gemm(p)) {
    return common::Status::Unsupported("v0 supports fp32 only");
  }

  // 指针转换与内核启动
  const auto* A = reinterpret_cast<const float*>(p.A);
  const auto* B = reinterpret_cast<const float*>(p.B);
  auto* C = reinterpret_cast<float*>(p.C);
  const dim3 block(16, 16);
  const dim3 grid((p.N + 15) / 16, (p.M + 15) / 16);
  gemm_gpu::GemmNaiveKernel<<<grid, block, 0, stream>>>(A, B, C, p.M, p.N, p.K);
  CKO_CUDA(cudaStreamSynchronize(stream));
  return common::Status::Ok();
}

common::Status RunV3(const GemmParams& p, cudaStream_t stream) {
  // 参数与对齐校验
  if (IsFp16Gemm(p)) {
    return common::Status::Unsupported("v3 fp32 kernel does not support fp16 inputs");
  }
  if (!IsAlignedForImpl(ImplId::kV3, p)) {
    return common::Status::InvalidArgument("shape not aligned for v3 kernel");
  }

  // 指针转换与 SMEM 配置
  const auto* A = reinterpret_cast<const float*>(p.A);
  const auto* B = reinterpret_cast<const float*>(p.B);
  auto* C = reinterpret_cast<float*>(p.C);
  const int smem_bytes =
      static_cast<int>((2 * gemm_v3::kSmemABuf + 2 * gemm_v3::kSmemBBuf) * sizeof(float));
  CKO_CUDA(cudaFuncSetAttribute(gemm_v3::GemmV3Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                smem_bytes));

  // 内核启动
  const dim3 block(gemm_v3::kBlockThreadsX, gemm_v3::kBlockThreadsY);
  const dim3 grid((p.N + gemm_v3::kBlockN - 1) / gemm_v3::kBlockN,
                  (p.M + gemm_v3::kBlockM - 1) / gemm_v3::kBlockM);
  gemm_v3::GemmV3Kernel<<<grid, block, smem_bytes, stream>>>(A, B, C, p.M, p.N, p.K);
  CKO_CUDA(cudaStreamSynchronize(stream));
  return common::Status::Ok();
}

common::Status RunFp16(const GemmParams& p, cudaStream_t stream) {
  // 参数与对齐校验
  if (!IsFp16Gemm(p)) {
    return common::Status::Unsupported("fp16 kernel requires fp16 A/B and fp32 C");
  }
  if (!IsAlignedForImpl(ImplId::kFp16, p)) {
    return common::Status::InvalidArgument("shape not aligned for fp16 kernel");
  }

#if defined(CKO_GEMM_CUTLASS_FP16)
  // CUTLASS FP16 GEMM 路径
  const auto* A = reinterpret_cast<const cutlass::half_t*>(p.A);
  const auto* B = reinterpret_cast<const cutlass::half_t*>(p.B);
  auto* C = reinterpret_cast<float*>(p.C);
  const cutlass::Status st =
      gemm_cute::LaunchGemmFp16RowMajor(p.M, p.N, p.K, A, B, C, p.alpha, p.beta, stream);
  if (st != cutlass::Status::kSuccess) {
    return common::Status::CudaError(cutlassGetStatusString(st));
  }
  CKO_CUDA(cudaStreamSynchronize(stream));
  return common::Status::Ok();
#else
  // 内置 FP16 GEMM 内核路径
  static bool smem_attr_set = false;
  if (!smem_attr_set) {
    CKO_CUDA(cudaFuncSetAttribute(gemm_fp16::GemmFP16Kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(gemm_fp16::kSmemSize)));
    smem_attr_set = true;
  }
  const auto* A = reinterpret_cast<const __half*>(p.A);
  const auto* B = reinterpret_cast<const __half*>(p.B);
  auto* C = reinterpret_cast<float*>(p.C);
  const dim3 block(gemm_fp16::kBlockThreadsX, gemm_fp16::kBlockThreadsY);
  const dim3 grid((p.N + gemm_fp16::kBlockN - 1) / gemm_fp16::kBlockN,
                  (p.M + gemm_fp16::kBlockM - 1) / gemm_fp16::kBlockM);
  gemm_fp16::GemmFP16Kernel<<<grid, block, gemm_fp16::kSmemSize, stream>>>(A, B, C, p.M, p.N, p.K);
  CKO_CUDA(cudaStreamSynchronize(stream));
  return common::Status::Ok();
#endif
}

namespace {
cublasLtHandle_t CublasLtHandle() {
  static cublasLtHandle_t handle = nullptr;
  if (!handle) {
    if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) return nullptr;
  }
  return handle;
}

cudaDataType CudaDtypeFromCko(common::DType dt) {
  switch (dt) {
    case common::DType::kFp32:  return CUDA_R_32F;
    case common::DType::kFp16:  return CUDA_R_16F;
    case common::DType::kBf16:  return CUDA_R_16BF;
    case common::DType::kInt8:  return CUDA_R_8I;
    case common::DType::kInt32: return CUDA_R_32I;
    case common::DType::kFp8E4M3: return CUDA_R_8F_E4M3;
    case common::DType::kFp8E5M2: return CUDA_R_8F_E5M2;
    default: return CUDA_R_32F;
  }
}

cublasComputeType_t CublasLtComputeType(common::DType dt) {
  switch (dt) {
    case common::DType::kInt8:  return CUBLAS_COMPUTE_32I;
    case common::DType::kFp16:  return CUBLAS_COMPUTE_32F_FAST_16F;
    case common::DType::kBf16:  return CUBLAS_COMPUTE_32F_FAST_16BF;
    default: return CUBLAS_COMPUTE_32F;
  }
}

common::Status RunCublasLtMatmul(const GemmParams& p, cudaStream_t stream) {
  // 获取句柄与类型映射
  cublasLtHandle_t lt_handle = CublasLtHandle();
  if (!lt_handle) return common::Status::CudaError("cublasLtCreate failed");

  cudaDataType a_type = CudaDtypeFromCko(p.dtype_a);
  cudaDataType b_type = CudaDtypeFromCko(p.dtype_b);
  bool is_int8 = (p.dtype_a == common::DType::kInt8);
  cudaDataType c_type_cublas = is_int8 ? CUDA_R_32I : CUDA_R_32F;
  cublasComputeType_t compute_type = CublasLtComputeType(p.dtype_a);

  // 设置缩放因子与指针类型
  float alpha = p.alpha;
  float beta = p.beta;
  int32_t alpha_i32 = static_cast<int32_t>(p.alpha);
  int32_t beta_i32 = static_cast<int32_t>(p.beta);
  void* alpha_ptr = is_int8 ? reinterpret_cast<void*>(&alpha_i32) : reinterpret_cast<void*>(&alpha);
  void* beta_ptr = is_int8 ? reinterpret_cast<void*>(&beta_i32) : reinterpret_cast<void*>(&beta);
  cudaDataType scale_type = is_int8 ? CUDA_R_32I : CUDA_R_32F;

  cublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr;
  cublasLtMatmulDesc_t matmul_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;

  auto cleanup = [&]() {
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (matmul_desc) cublasLtMatmulDescDestroy(matmul_desc);
    if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
    if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
    if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
  };

  #define CKOLT(call)                                         \
    do {                                                      \
      cublasStatus_t _s = (call);                             \
      if (_s != CUBLAS_STATUS_SUCCESS) {                      \
        cleanup();                                            \
        return common::Status::CudaError("cublasLt call failed"); \
      }                                                       \
    } while (0)

  // 创建矩阵布局描述符
  const int32_t order = CUBLASLT_ORDER_ROW;
  CKOLT(cublasLtMatrixLayoutCreate(&a_desc, a_type, p.M, p.K, p.lda));
  CKOLT(cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                          &order, sizeof(order)));
  CKOLT(cublasLtMatrixLayoutCreate(&b_desc, b_type, p.K, p.N, p.ldb));
  CKOLT(cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                          &order, sizeof(order)));
  CKOLT(cublasLtMatrixLayoutCreate(&c_desc, c_type_cublas, p.M, p.N, p.ldc));
  CKOLT(cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                          &order, sizeof(order)));

  // 创建矩阵乘法描述符与偏好设置
  CKOLT(cublasLtMatmulDescCreate(&matmul_desc, compute_type, scale_type));
  const cublasOperation_t trans_op = CUBLAS_OP_N;
  CKOLT(cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_op, sizeof(trans_op)));
  CKOLT(cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_op, sizeof(trans_op)));

  CKOLT(cublasLtMatmulPreferenceCreate(&pref));

  // 执行矩阵乘法
  cublasStatus_t st = cublasLtMatmul(lt_handle, matmul_desc,
                                     alpha_ptr, p.A, a_desc, p.B, b_desc,
                                     beta_ptr, p.C, c_desc, p.C, c_desc,
                                     nullptr, nullptr, 0, stream);
  cleanup();
  if (st != CUBLAS_STATUS_SUCCESS) {
    return common::Status::CudaError("cublasLtMatmul failed");
  }
  CKO_CUDA(cudaStreamSynchronize(stream));
  return common::Status::Ok();
  #undef CKOLT
}

}  // namespace

common::Status RunCublasFp32(const GemmParams& p, cudaStream_t stream) {
  if (IsFp16Gemm(p)) {
    return common::Status::Unsupported("use cublas fp16 path for fp16 inputs");
  }
  return RunCublasLtMatmul(p, stream);
}

common::Status RunCublasFp16(const GemmParams& p, cudaStream_t stream) {
  if (!IsFp16Gemm(p)) {
    return common::Status::Unsupported("cublas fp16 path requires fp16 A/B and fp32 C");
  }
  return RunCublasLtMatmul(p, stream);
}

common::Status RunCublasBf16(const GemmParams& p, cudaStream_t stream) {
  if (!IsBf16Gemm(p)) {
    return common::Status::Unsupported("cublas bf16 path requires bf16 A/B and fp32 C");
  }
  return RunCublasLtMatmul(p, stream);
}

common::Status RunCublasInt8(const GemmParams& p, cudaStream_t stream) {
  if (!IsInt8Gemm(p)) {
    return common::Status::Unsupported("cublas int8 path requires int8 A/B and int32 or fp32 C");
  }
  return RunCublasLtMatmul(p, stream);
}

common::Status RunCublasFp8(const GemmParams& p, cudaStream_t stream) {
  if (!IsFp8Gemm(p)) {
    return common::Status::Unsupported("cublas fp8 path requires fp8 A/B and fp32 C");
  }
  return RunCublasLtMatmul(p, stream);
}

#undef CKO_CUDA

}  // namespace

namespace {

cublasHandle_t FallbackHandle() {
  static cublasHandle_t h = nullptr;
  if (!h) {
    cublasCreate(&h);
  }
  return h;
}

template <typename T>
__global__ void StridedConvertToFp32Typed(const T* in, float* out,
                                          int rows, int cols, int ld_in) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || col >= cols) return;
  out[row * cols + col] = static_cast<float>(in[row * ld_in + col]);
}

template <typename T>
void LaunchStridedConvertToFp32(const void* d_in, float* d_out,
                                int rows, int cols, int ld_in,
                                cudaStream_t stream) {
  const dim3 block(256);
  const dim3 grid((cols + 255) / 256, rows);
  StridedConvertToFp32Typed<<<grid, block, 0, stream>>>(
      reinterpret_cast<const T*>(d_in), d_out, rows, cols, ld_in);
}

common::Status FallbackFp32Gemm(const GemmParams& p, cudaStream_t stream) {
  const int M = p.M, N = p.N, K = p.K;
  const int lda_src = p.lda;
  const int ldb_src = p.ldb;
  const size_t n_a_flat = static_cast<size_t>(M) * K;
  const size_t n_b_flat = static_cast<size_t>(K) * N;

  // 分配 FP32 临时缓冲区
  float *d_A = nullptr, *d_B = nullptr;
  if (cudaMalloc(&d_A, n_a_flat * sizeof(float)) != cudaSuccess) {
    return common::Status::CudaError("fallback: cudaMalloc fp32_A");
  }
  if (cudaMalloc(&d_B, n_b_flat * sizeof(float)) != cudaSuccess) {
    cudaFree(d_A);
    return common::Status::CudaError("fallback: cudaMalloc fp32_B");
  }

  // 转换 A 矩阵到 FP32
  if (lda_src > K) {
    switch (p.dtype_a) {
      case common::DType::kInt8:
        LaunchStridedConvertToFp32<int8_t>(p.A, d_A, M, K, lda_src, stream); break;
      case common::DType::kFp16:
        LaunchStridedConvertToFp32<__half>(p.A, d_A, M, K, lda_src, stream); break;
      case common::DType::kBf16:
        LaunchStridedConvertToFp32<__nv_bfloat16>(p.A, d_A, M, K, lda_src, stream); break;
      case common::DType::kFp8E4M3:
        LaunchStridedConvertToFp32<__nv_fp8_e4m3>(p.A, d_A, M, K, lda_src, stream); break;
      case common::DType::kFp8E5M2:
        LaunchStridedConvertToFp32<__nv_fp8_e5m2>(p.A, d_A, M, K, lda_src, stream); break;
      default: break;
    }
  } else {
    LaunchConvertToFp32(p.A, d_A, static_cast<int>(n_a_flat), p.dtype_a, stream);
  }

  // 转换 B 矩阵到 FP32
  if (ldb_src > N) {
    switch (p.dtype_b) {
      case common::DType::kInt8:
        LaunchStridedConvertToFp32<int8_t>(p.B, d_B, K, N, ldb_src, stream); break;
      case common::DType::kFp16:
        LaunchStridedConvertToFp32<__half>(p.B, d_B, K, N, ldb_src, stream); break;
      case common::DType::kBf16:
        LaunchStridedConvertToFp32<__nv_bfloat16>(p.B, d_B, K, N, ldb_src, stream); break;
      case common::DType::kFp8E4M3:
        LaunchStridedConvertToFp32<__nv_fp8_e4m3>(p.B, d_B, K, N, ldb_src, stream); break;
      case common::DType::kFp8E5M2:
        LaunchStridedConvertToFp32<__nv_fp8_e5m2>(p.B, d_B, K, N, ldb_src, stream); break;
      default: break;
    }
  } else {
    LaunchConvertToFp32(p.B, d_B, static_cast<int>(n_b_flat), p.dtype_b, stream);
  }

  // 执行 cuBLAS SGEMM
  cublasHandle_t h = FallbackHandle();
  cublasSetStream(h, stream);

  const float alpha = p.alpha;
  const float beta = p.beta;
  const cublasStatus_t cs = cublasSgemm(
      h, CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,
      &alpha,
      d_B, N,
      d_A, K,
      &beta,
      reinterpret_cast<float*>(const_cast<void*>(p.C)), p.ldc);

  // 清理并检查结果
  cudaFree(d_A);
  cudaFree(d_B);
  if (cs != CUBLAS_STATUS_SUCCESS) {
    return common::Status::CudaError("fallback cublasSgemm failed");
  }
  cudaStreamSynchronize(stream);
  return common::Status::Ok();
}

}  // namespace

bool IsAlignedForImpl(ImplId impl, const GemmParams& p) {
  switch (impl) {
    case ImplId::kV3:
      return (p.M % gemm_v3::kBlockM == 0) && (p.N % gemm_v3::kBlockN == 0) &&
             (p.K % gemm_v3::kTileK == 0);
    case ImplId::kV4:
    case ImplId::kFp16:
      return (p.M % gemm_fp16::kBlockM == 0) && (p.N % gemm_fp16::kBlockN == 0) &&
             (p.K % gemm_fp16::kTileK == 0);
    case ImplId::kCublasBf16:
    case ImplId::kCublasInt8:
    case ImplId::kCublasFp8:
      return true;
    default:
      return true;
  }
}

common::Status GemmRunGpu(const GemmParams& p, cudaStream_t stream) {
  // 布局检查
  if (p.layout != common::Layout::kRowMajor) {
    return common::Status::Unsupported("GemmRunGpu supports row_major only");
  }

  // 按实现类型分发
  switch (p.impl) {
    case ImplId::kV0:
      return RunV0(p, stream);
    case ImplId::kV3:
      return RunV3(p, stream);
    case ImplId::kFp16:
      return RunFp16(p, stream);
    case ImplId::kCublas:
      return RunCublasFp32(p, stream);
    case ImplId::kCublasFp16:
    case ImplId::kCublasBf16:
    case ImplId::kCublasInt8:
    case ImplId::kCublasFp8: {
      common::Status st;
      switch (p.impl) {
        case ImplId::kCublasFp16: st = RunCublasFp16(p, stream); break;
        case ImplId::kCublasBf16: st = RunCublasBf16(p, stream); break;
        case ImplId::kCublasInt8: st = RunCublasInt8(p, stream); break;
        case ImplId::kCublasFp8:  st = RunCublasFp8(p, stream); break;
        default: break;
      }
      if (!st.ok()) {
        return FallbackFp32Gemm(p, stream);
      }
      return st;
    }
    case ImplId::kV1:
    case ImplId::kV2:
    case ImplId::kV4:
      return common::Status::Unimplemented("GEMM impl not implemented");
    case ImplId::kAuto:
      break;
  }
  return common::Status::InvalidArgument("unknown impl id");
}

}  // namespace gemm
