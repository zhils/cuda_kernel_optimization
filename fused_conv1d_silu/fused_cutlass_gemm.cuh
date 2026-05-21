#pragma once

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

#include <cstdio>
#include <cstdlib>

namespace fused_cutlass {

using RowMajor = cutlass::layout::RowMajor;
using ColMajor = cutlass::layout::ColumnMajor;

using Sgemm = cutlass::gemm::device::Gemm<float, RowMajor, float, ColMajor, float, RowMajor>;

inline void CheckCutlass(cutlass::Status status, const char* msg) {
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "CUTLASS error: %s (status=%d)\n", msg, static_cast<int>(status));
    std::exit(EXIT_FAILURE);
  }
}

// C = A(M,K) @ W(N,K)^T，矩阵均为行主序。
// 等价于 B = W^T 作为列主序 (K,N)，ldb = K。
inline cutlass::Status GemmRowMajorWT(int M, int N, int K,
                                      const float* A, int lda,
                                      const float* W, int ldw,
                                      float* C, int ldc,
                                      float alpha = 1.0f, float beta = 0.0f,
                                      cudaStream_t stream = nullptr,
                                      void* workspace = nullptr,
                                      size_t workspace_bytes = 0) {
  Sgemm gemm;
  Sgemm::Arguments args({M, N, K},
                         {A, lda},
                         {W, ldw},
                         {C, ldc},
                         {C, ldc},
                         {alpha, beta});

  if (workspace != nullptr && workspace_bytes > 0) {
    CheckCutlass(gemm.can_implement(args), "can_implement");
    CheckCutlass(gemm.initialize(args, workspace, stream), "initialize");
    return gemm(stream);
  }

  return gemm(args, workspace, stream);
}

// CUTLASS GEMM 工作空间管理器，按需分配和扩容
struct GemmWorkspace {
  void* ptr = nullptr;
  size_t bytes = 0;

  void Ensure(Sgemm& gemm, const Sgemm::Arguments& args) {
    const size_t need = gemm.get_workspace_size(args);
    if (need <= bytes) return;
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
      bytes = 0;
    }
    if (need > 0) {
      if (cudaMalloc(&ptr, need) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc workspace failed\n");
        std::exit(EXIT_FAILURE);
      }
      bytes = need;
    }
  }

  ~GemmWorkspace() {
    if (ptr) cudaFree(ptr);
  }
};

// 带工作空间管理的高层 GEMM 启动接口
inline void LaunchGemmRowMajorWT(GemmWorkspace& ws, int M, int N, int K,
                                 const float* A, int lda,
                                 const float* W, int ldw,
                                 float* C, int ldc,
                                 float alpha = 1.0f, float beta = 0.0f,
                                 cudaStream_t stream = nullptr) {
  Sgemm gemm;
  Sgemm::Arguments args({M, N, K},
                        {A, lda},
                        {W, ldw},
                        {C, ldc},
                        {C, ldc},
                        {alpha, beta});
  ws.Ensure(gemm, args);
  CheckCutlass(gemm.can_implement(args), "can_implement");
  if (ws.bytes > 0) {
    CheckCutlass(gemm.initialize(args, ws.ptr, stream), "initialize");
    CheckCutlass(gemm(stream), "gemm");
  } else {
    CheckCutlass(gemm(args, nullptr, stream), "gemm");
  }
}

}  // namespace fused_cutlass
