#pragma once

#include "common/dtype.h"
#include "common/status.h"

#include <cuda_runtime.h>

namespace fused {

enum class ImplId : int {
  kAuto = 0,
  kV0 = 1,
  kV1 = 2,
  kV2 = 3,
  kV3 = 4,
};

struct FusedParams {
  int B = 0;
  int L = 0;
  int D = 0;
  int H = 0;
  int k_size = 0;
  common::DType dtype = common::DType::kFp32;
  const void* x = nullptr;
  const void* W_qkv = nullptr;
  const void* b_qkv = nullptr;
  const void* W_z = nullptr;
  const void* b_z = nullptr;
  const void* K_conv = nullptr;
  void* Q = nullptr;
  void* K = nullptr;
  void* V = nullptr;
  ImplId impl = ImplId::kAuto;
};

struct FusedTolerance {
  float atol = 1e-2f;
};

common::Status ValidateFusedParams(const FusedParams& p, bool require_device_ptrs = true);
common::Status FusedRun(const FusedParams& p, cudaStream_t stream = nullptr);

// Phase 2：宿主参考实现（见 fused_ref_cpu.h）。
common::Status FusedReferenceHost(const FusedParams& p);
float DefaultAbsTolerance();

}  // namespace fused

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CkoFusedParams {
  int B, L, D, H, k_size;
  int dtype;
  const void* x;
  const void* W_qkv;
  const void* b_qkv;
  const void* W_z;
  const void* b_z;
  const void* K_conv;
  void* Q;
  void* K;
  void* V;
  int impl;
} CkoFusedParams;

int cko_fused_conv1d_silu_validate(const CkoFusedParams* params, char* err_buf, size_t err_len);
int cko_fused_conv1d_silu_run(const CkoFusedParams* params, void* stream);

#ifdef __cplusplus
}
#endif
