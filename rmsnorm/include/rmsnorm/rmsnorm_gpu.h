#pragma once

#include "common/status.h"
#include "rmsnorm/rmsnorm_api.h"

#include <cuda_runtime.h>

namespace rmsnorm {

// Device pointers in RmsNormParams; launches selected impl on stream (default stream if null).
common::Status RmsNormRunGpu(const RmsNormParams& p, cudaStream_t stream = nullptr);

}  // namespace rmsnorm
