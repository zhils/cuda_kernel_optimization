// CUTLASS FP16 GEMM，基于 CuTe 布局，使用 Tensor Core，行主序 A/B，FP32 输出
#pragma once

#include <cstdio>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"

namespace gemm_cute {

// 瓦片尺寸常量
constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTileK = 32;

// 对齐检查
inline bool IsAligned(int M, int N, int K) {
  return (M % kTileM == 0) && (N % kTileN == 0) && (K % kTileK == 0);
}

namespace detail {

// 类型别名
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

// CUTLASS Gemm 配置
using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value, ElementAccumulator,
        ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

// 工作空间管理
struct Workspace {
  void* ptr = nullptr;
  size_t bytes = 0;

  void Ensure(Gemm& gemm, const typename Gemm::Arguments& args) {
    const size_t need = gemm.get_workspace_size(args);
    if (need <= bytes) return;
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
      bytes = 0;
    }
    if (need > 0) {
      cudaMalloc(&ptr, need);
      bytes = need;
    }
  }

  ~Workspace() {
    if (ptr) cudaFree(ptr);
  }
};

// 启动 CUTLASS GEMM
inline cutlass::Status LaunchGemm(int M, int N, int K, const ElementA* A, const ElementB* B,
                                  ElementC* C, float alpha, float beta, cudaStream_t stream,
                                  Workspace& ws) {
  Gemm gemm;
  typename Gemm::Arguments args({M, N, K}, {A, K}, {B, N}, {C, N}, {C, N}, {alpha, beta});
  cutlass::Status st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) return st;
  ws.Ensure(gemm, args);
  st = gemm.initialize(args, ws.ptr, stream);
  if (st != cutlass::Status::kSuccess) return st;
  return gemm(stream);
}

}  // namespace detail

// 对外接口：行主序 FP16 GEMM
inline cutlass::Status LaunchGemmFp16RowMajor(int M, int N, int K, const cutlass::half_t* A,
                                              const cutlass::half_t* B, float* C, float alpha,
                                              float beta, cudaStream_t stream) {
  static detail::Workspace ws;
  const cutlass::Status st =
      detail::LaunchGemm(M, N, K, A, B, C, alpha, beta, stream, ws);
  if (st != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "gemm_cute LaunchGemm failed: %s\n", cutlassGetStatusString(st));
  }
  return st;
}

}  // namespace gemm_cute
