#include "fused/fused_gpu.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "fused_cutlass_gemm.cuh"
#include "fused_kernels.cuh"

namespace fused {
namespace {

#define CKO_CUDA(call)                                                         \
  do {                                                                         \
    const cudaError_t err__ = (call);                                          \
    if (err__ != cudaSuccess) {                                                \
      return common::Status::CudaError(std::string(cudaGetErrorString(err__))); \
    }                                                                          \
  } while (0)

ImplId ResolveImpl(const FusedParams& p) {
  if (p.impl != ImplId::kAuto) return p.impl;
  return ImplId::kV3;
}

struct DeviceScratch {
  float* qkv = nullptr;
  float* z_proj = nullptr;
  float* z_act = nullptr;

  ~DeviceScratch() {
    if (qkv) cudaFree(qkv);
    if (z_proj) cudaFree(z_proj);
    if (z_act) cudaFree(z_act);
  }
};

common::Status AllocScratch(DeviceScratch& s, int BL, int H, bool need_qkv, bool need_z,
                            bool need_z_act) {
  if (need_qkv) {
    CKO_CUDA(cudaMalloc(&s.qkv, static_cast<size_t>(BL) * (3 * H) * sizeof(float)));
  }
  if (need_z) {
    CKO_CUDA(cudaMalloc(&s.z_proj, static_cast<size_t>(BL) * H * sizeof(float)));
  }
  if (need_z_act) {
    CKO_CUDA(cudaMalloc(&s.z_act, static_cast<size_t>(BL) * H * sizeof(float)));
  }
  return common::Status::Ok();
}

common::Status LaunchCutlassGemm(int M, int N, int K, const float* A, int lda, const float* W,
                                 int ldw, float* C, int ldc, float alpha, float beta,
                                 cudaStream_t stream, fused_cutlass::GemmWorkspace& ws) {
  using Sgemm = fused_cutlass::Sgemm;
  Sgemm gemm;
  Sgemm::Arguments args({M, N, K}, {A, lda}, {W, ldw}, {C, ldc}, {C, ldc}, {alpha, beta});
  ws.Ensure(gemm, args);
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
    return common::Status::Unsupported("CUTLASS gemm can_implement failed");
  }
  cutlass::Status st;
  if (ws.bytes > 0) {
    st = gemm.initialize(args, ws.ptr, stream);
    if (st != cutlass::Status::kSuccess) {
      return common::Status::CudaError("CUTLASS gemm initialize failed");
    }
    st = gemm(stream);
  } else {
    st = gemm(args, nullptr, stream);
  }
  if (st != cutlass::Status::kSuccess) {
    return common::Status::CudaError("CUTLASS gemm launch failed");
  }
  return common::Status::Ok();
}

common::Status RunV0(const FusedParams& p, cudaStream_t stream) {
  const int B = p.B;
  const int L = p.L;
  const int D = p.D;
  const int H = p.H;
  const int BL = B * L;
  const auto* x = reinterpret_cast<const float*>(p.x);
  const auto* W_qkv = reinterpret_cast<const float*>(p.W_qkv);
  const auto* b_qkv = reinterpret_cast<const float*>(p.b_qkv);
  const auto* W_z = reinterpret_cast<const float*>(p.W_z);
  const auto* b_z = reinterpret_cast<const float*>(p.b_z);
  const auto* K_conv = reinterpret_cast<const float*>(p.K_conv);
  auto* Q = reinterpret_cast<float*>(p.Q);
  auto* K = reinterpret_cast<float*>(p.K);
  auto* V = reinterpret_cast<float*>(p.V);

  DeviceScratch scratch;
  common::Status st = AllocScratch(scratch, BL, H, true, true, true);
  if (!st.ok()) return st;

  const dim3 block_lin(fused_gpu::kBlockSize);
  const dim3 grid_qkv(B, L, (3 * H + fused_gpu::kBlockSize - 1) / fused_gpu::kBlockSize);
  fused_gpu::LinearKernel<<<grid_qkv, block_lin, 0, stream>>>(x, W_qkv, b_qkv, scratch.qkv, B, L,
                                                                D, 3 * H);

  const dim3 grid_gate(B, L, (H + fused_gpu::kBlockSize - 1) / fused_gpu::kBlockSize);
  fused_gpu::LinearKernel<<<grid_gate, block_lin, 0, stream>>>(x, W_z, b_z, scratch.z_proj, B, L,
                                                                 D, H);

  const int total = BL * H;
  const int grid_split = (total + fused_gpu::kBlockSize - 1) / fused_gpu::kBlockSize;
  fused_gpu::SplitQKVKernel<<<grid_split, fused_gpu::kBlockSize, 0, stream>>>(
      scratch.qkv, Q, K, V, B, L, H);

  const int grid_conv = (total + fused_gpu::kBlockSize - 1) / fused_gpu::kBlockSize;
  fused_gpu::CausalConv1dSiLUKernel<<<grid_conv, fused_gpu::kBlockSize, 0, stream>>>(
      scratch.z_proj, K_conv, scratch.z_act, B, L, H, p.k_size);
  fused_gpu::GateMulKernel<<<grid_conv, fused_gpu::kBlockSize, 0, stream>>>(V, scratch.z_act, V,
                                                                            total);
  CKO_CUDA(cudaStreamSynchronize(stream));
  return common::Status::Ok();
}

common::Status RunV1(const FusedParams& p, cudaStream_t stream) {
  if (p.k_size > fused_gpu::kMaxKernelSize) {
    return common::Status::Unsupported("v1 supports k_size <= 64");
  }
  const dim3 block(fused_gpu::kBlockSize);
  const dim3 grid(p.B, (p.H + fused_gpu::kBlockSize - 1) / fused_gpu::kBlockSize);
  fused_gpu::FusedV1Kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const float*>(p.x), reinterpret_cast<const float*>(p.W_qkv),
      reinterpret_cast<const float*>(p.b_qkv), reinterpret_cast<const float*>(p.W_z),
      reinterpret_cast<const float*>(p.b_z), reinterpret_cast<const float*>(p.K_conv),
      reinterpret_cast<float*>(p.Q), reinterpret_cast<float*>(p.K), reinterpret_cast<float*>(p.V),
      p.B, p.L, p.D, p.H, p.k_size);
  const cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return common::Status::CudaError(cudaGetErrorString(err));
  }
  return common::Status::Ok();
}

common::Status RunV2(const FusedParams& p, cudaStream_t stream) {
  const int BL = p.B * p.L;
  DeviceScratch scratch;
  common::Status st = AllocScratch(scratch, BL, p.H, false, true, false);
  if (!st.ok()) return st;

  const int grid = (BL * p.H + fused_gpu::kBlockSize - 1) / fused_gpu::kBlockSize;
  fused_gpu::ComputeQKVZKernel<<<grid, fused_gpu::kBlockSize, 0, stream>>>(
      reinterpret_cast<const float*>(p.x), reinterpret_cast<const float*>(p.W_qkv),
      reinterpret_cast<const float*>(p.b_qkv), reinterpret_cast<const float*>(p.W_z),
      reinterpret_cast<const float*>(p.b_z), reinterpret_cast<float*>(p.Q),
      reinterpret_cast<float*>(p.K), reinterpret_cast<float*>(p.V), scratch.z_proj, p.B, p.L, p.D,
      p.H);
  fused_gpu::ConvGateKernel<<<grid, fused_gpu::kBlockSize, 0, stream>>>(
      scratch.z_proj, reinterpret_cast<const float*>(p.K_conv), reinterpret_cast<float*>(p.V), p.B,
      p.L, p.H, p.k_size);
  CKO_CUDA(cudaStreamSynchronize(stream));
  return common::Status::Ok();
}

common::Status RunV3(const FusedParams& p, cudaStream_t stream) {
  static fused_cutlass::GemmWorkspace gemm_ws;
  const int BL = p.B * p.L;
  const int H = p.H;
  const int D = p.D;

  DeviceScratch scratch;
  common::Status st = AllocScratch(scratch, BL, H, true, true, false);
  if (!st.ok()) return st;

  const auto* x = reinterpret_cast<const float*>(p.x);
  const auto* W_qkv = reinterpret_cast<const float*>(p.W_qkv);
  const auto* b_qkv = reinterpret_cast<const float*>(p.b_qkv);
  const auto* W_z = reinterpret_cast<const float*>(p.W_z);
  const auto* b_z = reinterpret_cast<const float*>(p.b_z);
  auto* Q = reinterpret_cast<float*>(p.Q);
  auto* K = reinterpret_cast<float*>(p.K);
  auto* V = reinterpret_cast<float*>(p.V);

  st = LaunchCutlassGemm(BL, 3 * H, D, x, D, W_qkv, D, scratch.qkv, 3 * H, 1.f, 0.f, stream,
                         gemm_ws);
  if (!st.ok()) return st;

  const int split_grid = (BL * H + fused_gpu::kBlockSize - 1) / fused_gpu::kBlockSize;
  fused_gpu::SplitQKVAddBiasKernel<<<split_grid, fused_gpu::kBlockSize, 0, stream>>>(
      scratch.qkv, b_qkv, Q, K, V, BL, H);

  st = LaunchCutlassGemm(BL, H, D, x, D, W_z, D, scratch.z_proj, H, 1.f, 0.f, stream, gemm_ws);
  if (!st.ok()) return st;

  fused_gpu::AddRowBiasKernel<<<split_grid, fused_gpu::kBlockSize, 0, stream>>>(scratch.z_proj,
                                                                                b_z, BL, H);

  const int total = p.B * p.L * H;
  const int grid = (total + fused_gpu::kBlockSize - 1) / fused_gpu::kBlockSize;
  fused_gpu::ConvGateKernel<<<grid, fused_gpu::kBlockSize, 0, stream>>>(
      scratch.z_proj, reinterpret_cast<const float*>(p.K_conv), V, p.B, p.L, H, p.k_size);
  CKO_CUDA(cudaStreamSynchronize(stream));
  return common::Status::Ok();
}

#undef CKO_CUDA

}  // namespace

common::Status FusedRunGpuFp32(const FusedParams& p, cudaStream_t stream);
common::Status FusedRunGpuFp16(const FusedParams& p, cudaStream_t stream);
common::Status FusedRunGpuBf16(const FusedParams& p, cudaStream_t stream);
common::Status FusedRunGpuInt8(const FusedParams& p, cudaStream_t stream);
common::Status FusedRunGpuFp8E4M3(const FusedParams& p, cudaStream_t stream);
common::Status FusedRunGpuFp8E5M2(const FusedParams& p, cudaStream_t stream);

common::Status FusedRunGpu(const FusedParams& p, cudaStream_t stream) {
  if (p.dtype == common::DType::kFp32) {
    return FusedRunGpuFp32(p, stream);
  }
  if (p.dtype == common::DType::kFp16) {
    return FusedRunGpuFp16(p, stream);
  }
  if (p.dtype == common::DType::kBf16) {
    return FusedRunGpuBf16(p, stream);
  }
  if (p.dtype == common::DType::kInt8) {
    return FusedRunGpuInt8(p, stream);
  }
  if (p.dtype == common::DType::kFp8E4M3) {
    return FusedRunGpuFp8E4M3(p, stream);
  }
  if (p.dtype == common::DType::kFp8E5M2) {
    return FusedRunGpuFp8E5M2(p, stream);
  }
  return common::Status::Unsupported("FusedRunGpu: unknown dtype");
}

common::Status FusedRunGpuFp32(const FusedParams& p, cudaStream_t stream) {
  const ImplId impl = ResolveImpl(p);
  switch (impl) {
    case ImplId::kV0:
      return RunV0(p, stream);
    case ImplId::kV1:
      return RunV1(p, stream);
    case ImplId::kV2:
      return RunV2(p, stream);
    case ImplId::kV3:
    case ImplId::kAuto:
      return RunV3(p, stream);
  }
  return common::Status::InvalidArgument("unknown impl id");
}

namespace {

cublasHandle_t GetCublasHandle() {
  static cublasHandle_t h = nullptr;
  if (!h) {
    cublasCreate(&h);
  }
  return h;
}

template <typename T>
struct CudaDtypeTraits;
template <> struct CudaDtypeTraits<__half> {
  static constexpr cudaDataType_t type = CUDA_R_16F;
  static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32F_FAST_16F;
};
template <> struct CudaDtypeTraits<__nv_bfloat16> {
  static constexpr cudaDataType_t type = CUDA_R_16BF;
  static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32F_FAST_16BF;
};
template <> struct CudaDtypeTraits<__nv_fp8_e4m3> {
  static constexpr cudaDataType_t type = CUDA_R_8F_E4M3;
  static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
};
template <> struct CudaDtypeTraits<__nv_fp8_e5m2> {
  static constexpr cudaDataType_t type = CUDA_R_8F_E5M2;
  static constexpr cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
};

template <typename T>
common::Status LaunchTypedGemm(int M, int N, int K,
                               const T* A, int lda,
                               const T* B, int ldb,
                               float* C, int ldc,
                               cudaStream_t stream) {
  cublasHandle_t h = GetCublasHandle();
  if (!h) return common::Status::CudaError("cublasCreate failed");
  cublasSetStream(h, stream);

  const float alpha = 1.f;
  const float beta = 0.f;
  const cublasStatus_t st = cublasGemmEx(
      h, CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,
      &alpha,
      B, CudaDtypeTraits<T>::type, ldb,
      A, CudaDtypeTraits<T>::type, lda,
      &beta,
      C, CUDA_R_32F, ldc,
      CudaDtypeTraits<T>::compute,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  if (st != CUBLAS_STATUS_SUCCESS) {
    return common::Status::CudaError("cublasGemmEx failed");
  }
  return common::Status::Ok();
}

template <>
common::Status LaunchTypedGemm<int8_t>(int M, int N, int K,
                                       const int8_t* A, int lda,
                                       const int8_t* B, int ldb,
                                       float* C, int ldc,
                                       cudaStream_t stream) {
  cublasHandle_t h = GetCublasHandle();
  if (!h) return common::Status::CudaError("cublasCreate failed");
  cublasSetStream(h, stream);

  int32_t* d_tmp = nullptr;
  const size_t out_size = static_cast<size_t>(M) * N * sizeof(int32_t);
  cudaError_t ce = cudaMalloc(&d_tmp, out_size);
  if (ce != cudaSuccess) return common::Status::CudaError("cudaMalloc int32 gemm scratch failed");

  const int32_t alpha = 1;
  const int32_t beta = 0;
  const cublasStatus_t st = cublasGemmEx(
      h, CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,
      &alpha,
      B, CUDA_R_8I, ldb,
      A, CUDA_R_8I, lda,
      &beta,
      d_tmp, CUDA_R_32I, ldc,
      CUBLAS_COMPUTE_32I,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  if (st != CUBLAS_STATUS_SUCCESS) {
    cudaFree(d_tmp);
    return common::Status::CudaError("cublasGemmEx int8 failed");
  }

  const int total = M * N;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  ConvertKernel<int32_t, float><<<blocks, threads, 0, stream>>>(d_tmp, C, total);
  cudaFree(d_tmp);
  return common::Status::Ok();
}

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080

namespace {

cublasLtHandle_t GetCublasLtHandle() {
  static cublasLtHandle_t h = nullptr;
  if (!h) {
    cublasLtCreate(&h);
  }
  return h;
}

}  // namespace

template <typename T>
common::Status LaunchTypedGemmFp8(cudaDataType_t a_type, int M, int N, int K,
                                  const T* A, int lda,
                                  const T* W, int ldw,
                                  float* C, int ldc,
                                  cudaStream_t stream) {
  cublasLtHandle_t h = GetCublasLtHandle();
  if (!h) return common::Status::CudaError("cublasLtCreate failed");

  const int32_t row_order = CUBLASLT_ORDER_ROW;
  const cublasOperation_t trans_n = CUBLAS_OP_N;
  const cublasOperation_t trans_t = CUBLAS_OP_T;

  cublasLtMatrixLayout_t a_desc = nullptr, w_desc = nullptr, c_desc = nullptr;
  cublasLtMatmulDesc_t matmul_desc = nullptr;

  auto cleanup = [&]() {
    if (matmul_desc) cublasLtMatmulDescDestroy(matmul_desc);
    if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
    if (w_desc) cublasLtMatrixLayoutDestroy(w_desc);
    if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
  };

  cublasStatus_t st = cublasLtMatrixLayoutCreate(&a_desc, a_type, M, K, lda);
  if (st != CUBLAS_STATUS_SUCCESS) { cleanup(); return common::Status::CudaError("a_desc"); }
  cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order));

  st = cublasLtMatrixLayoutCreate(&w_desc, a_type, N, K, ldw);
  if (st != CUBLAS_STATUS_SUCCESS) { cleanup(); return common::Status::CudaError("w_desc"); }
  cublasLtMatrixLayoutSetAttribute(w_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order));

  st = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, M, N, ldc);
  if (st != CUBLAS_STATUS_SUCCESS) { cleanup(); return common::Status::CudaError("c_desc"); }
  cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order));

  st = cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (st != CUBLAS_STATUS_SUCCESS) { cleanup(); return common::Status::CudaError("matmul_desc"); }
  cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_n, sizeof(trans_n));
  cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_t, sizeof(trans_t));

  const float alpha = 1.f;
  const float beta = 0.f;
  st = cublasLtMatmul(h, matmul_desc,
                      &alpha, A, a_desc, W, w_desc,
                      &beta, C, c_desc, C, c_desc,
                      nullptr, nullptr, 0, stream);
  cleanup();
  if (st != CUBLAS_STATUS_SUCCESS) {
    return common::Status::CudaError("cublasLtMatmul failed");
  }
  return common::Status::Ok();
}

template <>
common::Status LaunchTypedGemm<__nv_fp8_e4m3>(int M, int N, int K,
                                               const __nv_fp8_e4m3* A, int lda,
                                               const __nv_fp8_e4m3* B, int ldb,
                                               float* C, int ldc,
                                               cudaStream_t stream) {
  return LaunchTypedGemmFp8(CUDA_R_8F_E4M3, M, N, K, A, lda, B, ldb, C, ldc, stream);
}

template <>
common::Status LaunchTypedGemm<__nv_fp8_e5m2>(int M, int N, int K,
                                               const __nv_fp8_e5m2* A, int lda,
                                               const __nv_fp8_e5m2* B, int ldb,
                                               float* C, int ldc,
                                               cudaStream_t stream) {
  return LaunchTypedGemmFp8(CUDA_R_8F_E5M2, M, N, K, A, lda, B, ldb, C, ldc, stream);
}

#endif

template <typename T_W, bool kIsInt8>
struct TypedScratch {
  T_W* w_qkv = nullptr;
  T_W* w_z = nullptr;
  float* d_fp32 = nullptr;
  float* d_qkv = nullptr;
  float* d_z = nullptr;
  float* d_q = nullptr;
  float* d_k = nullptr;
  float* d_v = nullptr;

  common::Status Alloc(const FusedParams& p, cudaStream_t stream) {
    const int D = p.D;
    const int H = p.H;
    const int BL = p.B * p.L;
    const size_t n_w_qkv = static_cast<size_t>(3 * H) * D;
    const size_t n_w_z = static_cast<size_t>(H) * D;
    const size_t n_out = static_cast<size_t>(BL) * H;

    cudaError_t ce;
    ce = cudaMalloc(&w_qkv, n_w_qkv * sizeof(T_W));
    if (ce != cudaSuccess) return common::Status::CudaError("alloc w_qkv typed");
    ce = cudaMalloc(&w_z, n_w_z * sizeof(T_W));
    if (ce != cudaSuccess) { cudaFree(w_qkv); return common::Status::CudaError("alloc w_z typed"); }
    ce = cudaMalloc(&d_fp32, n_out * 7 * sizeof(float));
    if (ce != cudaSuccess) { cudaFree(w_qkv); cudaFree(w_z); return common::Status::CudaError("alloc fp32 scratch"); }
    d_qkv = d_fp32;
    d_z   = d_fp32 + n_out * 3;
    d_q   = d_fp32 + n_out * 4;
    d_k   = d_fp32 + n_out * 5;
    d_v   = d_fp32 + n_out * 6;
    (void)stream;
    return common::Status::Ok();
  }

  void Free() {
    cudaFree(w_qkv);
    cudaFree(w_z);
    cudaFree(d_fp32);
  }
};

template <typename T_W>
common::Status RunTypedImpl(const FusedParams& p, cudaStream_t stream) {
  constexpr bool kIsInt8 = std::is_same<T_W, int8_t>::value;
  const int BL = p.B * p.L;
  const int D = p.D;
  const int H = p.H;
  const int k_size = p.k_size;
  const int threads = 256;

  // 分配临时缓冲区
  TypedScratch<T_W, kIsInt8> s;
  common::Status st = s.Alloc(p, stream);
  if (!st.ok()) return st;

  // 权重类型转换与转置
  if constexpr (std::is_same<T_W, __nv_bfloat16>::value) {
    const int total_wq = 3 * H * D;
    const int blocks_wq = (total_wq + threads - 1) / threads;
    ConvertTransposeKernel<__nv_bfloat16><<<blocks_wq, threads, 0, stream>>>(
        reinterpret_cast<const float*>(p.W_qkv), 3 * H, D, D,
        s.w_qkv, 3 * H);
    const int total_wz = H * D;
    const int blocks_wz = (total_wz + threads - 1) / threads;
    ConvertTransposeKernel<__nv_bfloat16><<<blocks_wz, threads, 0, stream>>>(
        reinterpret_cast<const float*>(p.W_z), H, D, D,
        s.w_z, H);
  } else if constexpr (std::is_same<T_W, __half>::value) {
    const int total_wq = 3 * H * D;
    const int blocks_wq = (total_wq + threads - 1) / threads;
    ConvertTransposeKernel<__half><<<blocks_wq, threads, 0, stream>>>(
        reinterpret_cast<const float*>(p.W_qkv), 3 * H, D, D,
        reinterpret_cast<__half*>(s.w_qkv), 3 * H);
    const int total_wz = H * D;
    const int blocks_wz = (total_wz + threads - 1) / threads;
    ConvertTransposeKernel<__half><<<blocks_wz, threads, 0, stream>>>(
        reinterpret_cast<const float*>(p.W_z), H, D, D,
        reinterpret_cast<__half*>(s.w_z), H);
  } else if constexpr (std::is_same<T_W, int8_t>::value) {
    const int total_wq = 3 * H * D;
    const int blocks_wq = (total_wq + threads - 1) / threads;
    ConvertTransposeKernel<int8_t><<<blocks_wq, threads, 0, stream>>>(
        reinterpret_cast<const float*>(p.W_qkv), 3 * H, D, D,
        reinterpret_cast<int8_t*>(s.w_qkv), 3 * H);
    const int total_wz = H * D;
    const int blocks_wz = (total_wz + threads - 1) / threads;
    ConvertTransposeKernel<int8_t><<<blocks_wz, threads, 0, stream>>>(
        reinterpret_cast<const float*>(p.W_z), H, D, D,
        reinterpret_cast<int8_t*>(s.w_z), H);
  } else if constexpr (std::is_same<T_W, __nv_fp8_e4m3>::value) {
    const int total_wq = 3 * H * D;
    const int blocks_wq = (total_wq + threads - 1) / threads;
    ConvertKernel<float, __nv_fp8_e4m3><<<blocks_wq, threads, 0, stream>>>(
        reinterpret_cast<const float*>(p.W_qkv), s.w_qkv, total_wq);
    const int total_wz = H * D;
    const int blocks_wz = (total_wz + threads - 1) / threads;
    ConvertKernel<float, __nv_fp8_e4m3><<<blocks_wz, threads, 0, stream>>>(
        reinterpret_cast<const float*>(p.W_z), s.w_z, total_wz);
  } else if constexpr (std::is_same<T_W, __nv_fp8_e5m2>::value) {
    const int total_wq = 3 * H * D;
    const int blocks_wq = (total_wq + threads - 1) / threads;
    ConvertKernel<float, __nv_fp8_e5m2><<<blocks_wq, threads, 0, stream>>>(
        reinterpret_cast<const float*>(p.W_qkv), s.w_qkv, total_wq);
    const int total_wz = H * D;
    const int blocks_wz = (total_wz + threads - 1) / threads;
    ConvertKernel<float, __nv_fp8_e5m2><<<blocks_wz, threads, 0, stream>>>(
        reinterpret_cast<const float*>(p.W_z), s.w_z, total_wz);
  }

  // QKV 矩阵乘法
  const bool kIsFp8 = std::is_same<T_W, __nv_fp8_e4m3>::value || std::is_same<T_W, __nv_fp8_e5m2>::value;
  const int ld_w_qkv = kIsFp8 ? D : 3 * H;
  const int ld_w_z   = kIsFp8 ? D : H;

  st = LaunchTypedGemm<T_W>(BL, 3 * H, D,
                            reinterpret_cast<const T_W*>(p.x), D,
                            s.w_qkv, ld_w_qkv,
                            s.d_qkv, 3 * H,
                            stream);
  if (!st.ok()) { s.Free(); return st; }

  const int split_grid = (BL * H + fused_gpu::kBlockSize - 1) / fused_gpu::kBlockSize;
  fused_gpu::SplitQKVAddBiasKernel<<<split_grid, fused_gpu::kBlockSize, 0, stream>>>(
      s.d_qkv,
      reinterpret_cast<const float*>(p.b_qkv),
      s.d_q, s.d_k, s.d_v,
      BL, H);

  // Z 矩阵乘法
  st = LaunchTypedGemm<T_W>(BL, H, D,
                            reinterpret_cast<const T_W*>(p.x), D,
                            s.w_z, ld_w_z,
                            s.d_z, H,
                            stream);
  if (!st.ok()) { s.Free(); return st; }

  fused_gpu::AddRowBiasKernel<<<split_grid, fused_gpu::kBlockSize, 0, stream>>>(
      s.d_z, reinterpret_cast<const float*>(p.b_z), BL, H);

  // 卷积门控
  const int total = BL * H;
  const int grid = (total + fused_gpu::kBlockSize - 1) / fused_gpu::kBlockSize;
  fused_gpu::ConvGateKernel<<<grid, fused_gpu::kBlockSize, 0, stream>>>(
      s.d_z,
      reinterpret_cast<const float*>(p.K_conv),
      s.d_v,
      p.B, p.L, H, k_size);

  // 输出类型转换
  const int blocks_out = (total + threads - 1) / threads;
  if constexpr (std::is_same<T_W, __nv_bfloat16>::value) {
    ConvertKernel<float, __nv_bfloat16><<<blocks_out, threads, 0, stream>>>(
        s.d_q, reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(p.Q)), total);
    ConvertKernel<float, __nv_bfloat16><<<blocks_out, threads, 0, stream>>>(
        s.d_k, reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(p.K)), total);
    ConvertKernel<float, __nv_bfloat16><<<blocks_out, threads, 0, stream>>>(
        s.d_v, reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(p.V)), total);
  } else if constexpr (std::is_same<T_W, __half>::value) {
    ConvertKernel<float, __half><<<blocks_out, threads, 0, stream>>>(
        s.d_q, reinterpret_cast<__half*>(const_cast<void*>(p.Q)), total);
    ConvertKernel<float, __half><<<blocks_out, threads, 0, stream>>>(
        s.d_k, reinterpret_cast<__half*>(const_cast<void*>(p.K)), total);
    ConvertKernel<float, __half><<<blocks_out, threads, 0, stream>>>(
        s.d_v, reinterpret_cast<__half*>(const_cast<void*>(p.V)), total);
  } else if constexpr (std::is_same<T_W, int8_t>::value) {
    ConvertKernel<float, int8_t><<<blocks_out, threads, 0, stream>>>(
        s.d_q, reinterpret_cast<int8_t*>(const_cast<void*>(p.Q)), total);
    ConvertKernel<float, int8_t><<<blocks_out, threads, 0, stream>>>(
        s.d_k, reinterpret_cast<int8_t*>(const_cast<void*>(p.K)), total);
    ConvertKernel<float, int8_t><<<blocks_out, threads, 0, stream>>>(
        s.d_v, reinterpret_cast<int8_t*>(const_cast<void*>(p.V)), total);
  } else if constexpr (std::is_same<T_W, __nv_fp8_e4m3>::value) {
    ConvertKernel<float, __nv_fp8_e4m3><<<blocks_out, threads, 0, stream>>>(
        s.d_q, reinterpret_cast<__nv_fp8_e4m3*>(const_cast<void*>(p.Q)), total);
    ConvertKernel<float, __nv_fp8_e4m3><<<blocks_out, threads, 0, stream>>>(
        s.d_k, reinterpret_cast<__nv_fp8_e4m3*>(const_cast<void*>(p.K)), total);
    ConvertKernel<float, __nv_fp8_e4m3><<<blocks_out, threads, 0, stream>>>(
        s.d_v, reinterpret_cast<__nv_fp8_e4m3*>(const_cast<void*>(p.V)), total);
  } else if constexpr (std::is_same<T_W, __nv_fp8_e5m2>::value) {
    ConvertKernel<float, __nv_fp8_e5m2><<<blocks_out, threads, 0, stream>>>(
        s.d_q, reinterpret_cast<__nv_fp8_e5m2*>(const_cast<void*>(p.Q)), total);
    ConvertKernel<float, __nv_fp8_e5m2><<<blocks_out, threads, 0, stream>>>(
        s.d_k, reinterpret_cast<__nv_fp8_e5m2*>(const_cast<void*>(p.K)), total);
    ConvertKernel<float, __nv_fp8_e5m2><<<blocks_out, threads, 0, stream>>>(
        s.d_v, reinterpret_cast<__nv_fp8_e5m2*>(const_cast<void*>(p.V)), total);
  }

  cudaStreamSynchronize(stream);
  s.Free();
  return common::Status::Ok();
}

}  // namespace

common::Status FusedRunGpuBf16(const FusedParams& p, cudaStream_t stream) {
  return RunTypedImpl<__nv_bfloat16>(p, stream);
}

common::Status FusedRunGpuFp16(const FusedParams& p, cudaStream_t stream) {
  return RunTypedImpl<__half>(p, stream);
}

common::Status FusedRunGpuInt8(const FusedParams& p, cudaStream_t stream) {
  return RunTypedImpl<int8_t>(p, stream);
}

common::Status FusedRunGpuFp8E4M3(const FusedParams& p, cudaStream_t stream) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
  return RunTypedImpl<__nv_fp8_e4m3>(p, stream);
#else
  (void)p; (void)stream;
  return common::Status::Unsupported("FusedRunGpuFp8E4M3 requires CUDA >= 11.8");
#endif
}

common::Status FusedRunGpuFp8E5M2(const FusedParams& p, cudaStream_t stream) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
  common::Status st = RunTypedImpl<__nv_fp8_e5m2>(p, stream);
  if (st.ok()) return st;
  // FP8_E5M2 在当前 GPU 上不受支持，回退到 FP32 流水线
  const int BL = p.B * p.L;
  const size_t n_x = static_cast<size_t>(BL) * p.D;
  const size_t n_out = static_cast<size_t>(BL) * p.H;

  float *fp32_x = nullptr, *fp32_q = nullptr, *fp32_k = nullptr, *fp32_v = nullptr;
  cudaMalloc(&fp32_x, n_x * sizeof(float));
  cudaMalloc(&fp32_q, n_out * sizeof(float));
  cudaMalloc(&fp32_k, n_out * sizeof(float));
  cudaMalloc(&fp32_v, n_out * sizeof(float));

  const int threads = 256;
  const int blocks_x = (static_cast<int>(n_x) + threads - 1) / threads;
  const int blocks_out = (static_cast<int>(n_out) + threads - 1) / threads;
  ConvertKernel<__nv_fp8_e5m2, float><<<blocks_x, threads, 0, stream>>>(
      reinterpret_cast<const __nv_fp8_e5m2*>(p.x), fp32_x, static_cast<int>(n_x));

  FusedParams p_fp32 = p;
  p_fp32.x = fp32_x;
  p_fp32.Q = fp32_q;
  p_fp32.K = fp32_k;
  p_fp32.V = fp32_v;
  p_fp32.dtype = common::DType::kFp32;
  st = FusedRunGpuFp32(p_fp32, stream);

  if (st.ok()) {
    ConvertKernel<float, __nv_fp8_e5m2><<<blocks_out, threads, 0, stream>>>(
        fp32_q, reinterpret_cast<__nv_fp8_e5m2*>(const_cast<void*>(p.Q)), static_cast<int>(n_out));
    ConvertKernel<float, __nv_fp8_e5m2><<<blocks_out, threads, 0, stream>>>(
        fp32_k, reinterpret_cast<__nv_fp8_e5m2*>(const_cast<void*>(p.K)), static_cast<int>(n_out));
    ConvertKernel<float, __nv_fp8_e5m2><<<blocks_out, threads, 0, stream>>>(
        fp32_v, reinterpret_cast<__nv_fp8_e5m2*>(const_cast<void*>(p.V)), static_cast<int>(n_out));
  }

  cudaStreamSynchronize(stream);
  cudaFree(fp32_x);
  cudaFree(fp32_q);
  cudaFree(fp32_k);
  cudaFree(fp32_v);
  return st;
#else
  (void)p; (void)stream;
  return common::Status::Unsupported("FusedRunGpuFp8E5M2 requires CUDA >= 11.8");
#endif
}

}  // namespace fused
