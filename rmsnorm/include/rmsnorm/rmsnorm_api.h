#pragma once

#include "common/dtype.h"
#include "common/status.h"

#include <cuda_runtime.h>

namespace rmsnorm {

enum class ImplId : int {
  kAuto = 0,
  kV0 = 1,
  kV1 = 2,
  kV2 = 3,
  kV3 = 4,
  kCubRef = 5,
};

struct RmsNormParams {
  int rows = 0;
  int cols = 0;
  common::DType act_dtype = common::DType::kFp32;
  common::DType weight_dtype = common::DType::kFp32;
  float eps = 1e-5f;
  const void* input = nullptr;
  const void* weight = nullptr;
  void* output = nullptr;
  const float* input_scale = nullptr;
  float* output_scale = nullptr;
  ImplId impl = ImplId::kAuto;
};

struct RmsNormTolerance {
  float atol = 1e-2f;
  float rtol = 1e-2f;
};

common::Status ValidateRmsNormParams(const RmsNormParams& p, bool require_device_ptrs = true);
common::Status RmsNormRun(const RmsNormParams& p, cudaStream_t stream = nullptr);

// Phase 1：宿主参考实现（见 rmsnorm_ref_cpu.h）。
common::Status RmsNormReferenceHost(const RmsNormParams& p);
float DefaultAbsTolerance(common::DType act_dtype);

}  // namespace rmsnorm

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CkoRmsNormParams {
  int rows, cols;
  int act_dtype;
  int weight_dtype;
  float eps;
  const void* input;
  const void* weight;
  void* output;
  const float* input_scale;
  float* output_scale;
  int impl;
} CkoRmsNormParams;

int cko_rmsnorm_validate(const CkoRmsNormParams* params, char* err_buf, size_t err_len);
int cko_rmsnorm_run(const CkoRmsNormParams* params, void* stream);

#ifdef __cplusplus
}
#endif
