#pragma once

#include "common/dtype.h"
#include "common/status.h"

#include <cuda_runtime.h>

#include <string>

namespace gemm {

// 手写 kernel（v1–v4、fp16）要求 M/N/K 能被 tile 尺寸整除（128/128/32）。
// kFp16 在配置 CUTLASS 时使用 Sm80 Tensor Core。
// kFallback：不对齐时自动路由到 cuBLAS / naive（默认）。
// kStrict：若所选实现无法运行则返回 InvalidArgument。
// kSkip：返回 Skip 并附带消息（benchmark harness）。
enum class AlignmentPolicy : int {
  kFallback = 0,
  kStrict = 1,
  kSkip = 2,
};

enum class ImplId : int {
  kAuto = 0,
  kV0 = 1,
  kV1 = 2,
  kV2 = 3,
  kV3 = 4,
  kV4 = 5,
  kFp16 = 6,
  kCublas = 7,
  kCublasFp16 = 8,
  kCublasBf16 = 9,
  kCublasInt8 = 10,
  kCublasFp8 = 11,
};

struct GemmParams {
  int M = 0;
  int N = 0;
  int K = 0;
  common::DType dtype_a = common::DType::kFp32;
  common::DType dtype_b = common::DType::kFp32;
  common::DType dtype_c = common::DType::kFp32;
  common::Layout layout = common::Layout::kRowMajor;
  float alpha = 1.f;
  float beta = 0.f;
  const void* A = nullptr;
  const void* B = nullptr;
  void* C = nullptr;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  ImplId impl = ImplId::kAuto;
  AlignmentPolicy alignment_policy = AlignmentPolicy::kFallback;
};

struct GemmTolerance {
  float atol = 1e-3f;
  float rtol = 1e-3f;
};

// 手写 kernel 的 tile 要求（供调用方参考）。
struct GemmTileRequirements {
  int block_m = 128;
  int block_n = 128;
  int tile_k = 32;
};

common::Status ValidateGemmParams(const GemmParams& p, bool require_device_ptrs = true);
bool IsAlignedForTiledKernel(const GemmParams& p, const GemmTileRequirements& req);
ImplId ResolveImpl(const GemmParams& p, common::Status* alignment_note);

common::Status GemmRun(const GemmParams& p, cudaStream_t stream = nullptr);

}  // namespace gemm

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CkoGemmParams {
  int M, N, K;
  int dtype_a, dtype_b, dtype_c;
  int layout;
  float alpha, beta;
  const void* A;
  const void* B;
  void* C;
  int lda, ldb, ldc;
  int impl;
  int alignment_policy;
} CkoGemmParams;

int cko_gemm_validate(const CkoGemmParams* params, char* err_buf, size_t err_len);
int cko_gemm_run(const CkoGemmParams* params, void* stream);

#ifdef __cplusplus
}
#endif
