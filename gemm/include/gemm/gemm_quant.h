#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "common/dtype.h"
#include "common/status.h"

namespace gemm {

enum class QuantScheme {
  kPerTensor,
  kPerRow,
};

inline float QuantMax(common::DType dt) {
  switch (dt) {
    case common::DType::kInt8:     return 127.f;
    case common::DType::kFp8E4M3: return 448.f;
    case common::DType::kFp8E5M2: return 57344.f;
    default: return 1.f;
  }
}

void LaunchQuantizeMatrix(const float* d_in, void* d_out, float* d_scale,
                          int rows, int cols, common::DType dtype,
                          QuantScheme scheme, cudaStream_t stream);

void LaunchDequantizeGemmOutput(const void* d_in, float* d_out,
                                const float* d_scale_a, const float* d_scale_b,
                                int M, int N, common::DType compute_dtype,
                                QuantScheme scheme_a, QuantScheme scheme_b,
                                cudaStream_t stream);

common::Status GemmQuantizedRun(const float* d_A_fp32, const float* d_B_fp32,
                                float* d_C_fp32, int M, int N, int K,
                                common::DType quant_dtype, QuantScheme scheme_a,
                                QuantScheme scheme_b, cudaStream_t stream);

}  // namespace gemm
