#pragma once

#include "common/status.h"
#include "gemm/gemm_api.h"

#include <cuda_runtime.h>

namespace gemm {

common::Status GemmRunGpu(const GemmParams& p, cudaStream_t stream = nullptr);

bool IsAlignedForImpl(ImplId impl, const GemmParams& p);

}  // namespace gemm
