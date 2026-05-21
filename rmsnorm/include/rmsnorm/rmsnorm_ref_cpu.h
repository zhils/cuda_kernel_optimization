#pragma once

#include "common/dtype.h"
#include "common/status.h"
#include "rmsnorm/rmsnorm_api.h"

#include <vector>

namespace rmsnorm {

// Host-only reference (Phase 1). All pointers in RmsNormParams must be host-accessible.
common::Status RmsNormReferenceHost(const RmsNormParams& p);

// Match rmsnorm_v3 benchmark tolerances for GPU compare (Phase 2).
float DefaultAbsTolerance(common::DType act_dtype);

// Core FP32 math used by reference and tests.
void RmsNormFp32Core(const float* x, const float* weight, float* y, int rows, int cols,
                     float eps);

// Dequantize output matrix to FP32 for numerical checks (all activation dtypes).
common::Status DequantizeOutputToFp32(common::DType act_dtype, const void* output,
                                      const float* output_scale, int rows, int cols,
                                      std::vector<float>& out_fp32);

}  // namespace rmsnorm
