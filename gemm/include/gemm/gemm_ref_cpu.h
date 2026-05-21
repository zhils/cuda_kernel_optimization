#pragma once

#include "common/status.h"
#include "gemm/gemm_api.h"

namespace gemm {

common::Status GemmReferenceHost(const GemmParams& p);

float DefaultAbsTolerance(common::DType dtype_c);

void GemmFp32Core(const float* A, const float* B, float* C, int M, int N, int K, float alpha,
                  float beta);

}  // namespace gemm
