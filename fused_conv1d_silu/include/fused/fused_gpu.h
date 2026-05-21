#pragma once

#include "common/status.h"
#include "fused/fused_api.h"

#include <cuda_runtime.h>

namespace fused {

common::Status FusedRunGpu(const FusedParams& p, cudaStream_t stream = nullptr);

}  // namespace fused
