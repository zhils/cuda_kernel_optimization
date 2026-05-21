#include "gemm/gemm_ref_cpu.h"

#include <cuda_fp16.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gemm {

void GemmFp32Core(const float* A, const float* B, float* C, int M, int N, int K, float alpha,
                  float beta) {
#if defined(_OPENMP)
  #pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c) {
      float sum = 0.f;
      for (int k = 0; k < K; ++k) {
        sum += A[static_cast<size_t>(r) * K + k] * B[static_cast<size_t>(k) * N + c];
      }
      const size_t idx = static_cast<size_t>(r) * N + c;
      C[idx] = alpha * sum + beta * C[idx];
    }
  }
}

float DefaultAbsTolerance(common::DType dtype_c) {
  if (dtype_c == common::DType::kFp16) return 5e-2f;
  return 1e-3f;
}

common::Status GemmReferenceHost(const GemmParams& p) {
  common::Status st = ValidateGemmParams(p, false);
  if (!st.ok()) return st;
  if (!p.A || !p.B || !p.C) {
    return common::Status::InvalidArgument("A, B, C host pointers required");
  }
  if (p.layout != common::Layout::kRowMajor) {
    return common::Status::Unsupported("GemmReferenceHost supports row_major only");
  }
  if (p.dtype_a != common::DType::kFp32 || p.dtype_b != common::DType::kFp32 ||
      p.dtype_c != common::DType::kFp32) {
    return common::Status::Unsupported("GemmReferenceHost supports fp32 A/B/C only");
  }

  GemmFp32Core(reinterpret_cast<const float*>(p.A), reinterpret_cast<const float*>(p.B),
               reinterpret_cast<float*>(p.C), p.M, p.N, p.K, p.alpha, p.beta);
  return common::Status::Ok();
}

}  // namespace gemm
