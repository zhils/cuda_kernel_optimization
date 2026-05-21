#pragma once

#include "gemm/gemm_api.h"

namespace gemm {

enum class TuningStrategy {
  kMaxPerformance,
  kBalanced,
  kMaxPrecision,
};

inline const char* StrategyName(TuningStrategy s) {
  switch (s) {
    case TuningStrategy::kMaxPerformance: return "max_perf";
    case TuningStrategy::kBalanced:       return "balanced";
    case TuningStrategy::kMaxPrecision:   return "max_precision";
  }
  return "unknown";
}

struct TuningResult {
  common::DType dtype_a;
  common::DType dtype_b;
  common::DType dtype_c;
  ImplId impl;
  const char* reason;
};

inline bool IsFp8RuntimeAvailable() {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
  return true;
#else
  return false;
#endif
}

inline TuningResult TuneGemm(int M, int N, int K,
                             TuningStrategy strategy = TuningStrategy::kMaxPerformance) {
  TuningResult r;
  r.dtype_c = common::DType::kFp32;

  const bool aligned_8 = (M % 8 == 0) && (N % 8 == 0) && (K % 8 == 0);
  const bool aligned_4 = (M % 4 == 0) && (N % 4 == 0) && (K % 4 == 0);
  const bool aligned_v3 = (M % 128 == 0) && (N % 128 == 0) && (K % 32 == 0);
  const bool fp8_ok = IsFp8RuntimeAvailable();

  const auto nr_elements = static_cast<long long>(M) * N * K;
  const bool large_work = (nr_elements >= 16LL * 1024 * 1024);
  const bool modest_work = (nr_elements >= 1LL * 1024 * 1024);

  if (strategy == TuningStrategy::kMaxPrecision) {
    r.dtype_a = r.dtype_b = common::DType::kFp32;
    r.impl = ImplId::kCublas;
    r.reason = "max precision: fp32 cuBLAS";
    return r;
  }

  if (strategy == TuningStrategy::kBalanced) {
    if (K >= 512 && fp8_ok && aligned_8) {
      r.dtype_a = r.dtype_b = common::DType::kFp8E4M3;
      r.impl = ImplId::kCublasFp8;
      r.reason = "balanced: K>=512 8-aligned, fp8_e4m3 Tensor Core, high speed & precision";
    } else if (K >= 256 || (K >= 128 && large_work)) {
      r.dtype_a = r.dtype_b = common::DType::kBf16;
      r.impl = ImplId::kCublasBf16;
      r.reason = "balanced: bf16 Tensor Core, good speed & precision";
    } else if (K >= 128 && aligned_v3) {
      r.dtype_a = r.dtype_b = common::DType::kFp32;
      r.impl = ImplId::kV3;
      r.reason = "balanced: small shape, V3 hand-written FP32";
    } else {
      r.dtype_a = r.dtype_b = common::DType::kFp32;
      r.impl = ImplId::kCublas;
      r.reason = "balanced: small shape, fp32 cuBLAS";
    }
    return r;
  }

  if (K >= 512) {
    if (fp8_ok && aligned_8) {
      r.dtype_a = r.dtype_b = common::DType::kFp8E4M3;
      r.impl = ImplId::kCublasFp8;
      r.reason = "max_perf: K>=512 8-aligned, fp8_e4m3 Tensor Core (~12x fp32)";
    } else if (aligned_4) {
      r.dtype_a = r.dtype_b = common::DType::kInt8;
      r.impl = ImplId::kCublasInt8;
      r.reason = "max_perf: K>=512 4-aligned, int8 Tensor Core (~10x fp32)";
    } else {
      r.dtype_a = r.dtype_b = common::DType::kBf16;
      r.impl = ImplId::kCublasBf16;
      r.reason = "max_perf: K>=512 unaligned, bf16 Tensor Core (~3.7x fp32)";
    }
  } else if (K >= 256 || (K >= 128 && large_work)) {
    r.dtype_a = r.dtype_b = common::DType::kBf16;
    r.impl = ImplId::kCublasBf16;
    r.reason = "max_perf: bf16 Tensor Core (total work amortizes TC overhead)";
  } else if (K >= 128 && aligned_v3) {
    r.dtype_a = r.dtype_b = common::DType::kFp32;
    r.impl = ImplId::kV3;
    r.reason = "max_perf: small shape aligned, V3 hand-written FP32";
  } else if (K >= 64 && modest_work && aligned_8) {
    r.dtype_a = r.dtype_b = common::DType::kBf16;
    r.impl = ImplId::kCublasBf16;
    r.reason = "max_perf: K>=64 with enough work, bf16 bandwidth reduction (~1.3x)";
  } else {
    r.dtype_a = r.dtype_b = common::DType::kFp32;
    r.impl = ImplId::kCublas;
    r.reason = "max_perf: small shape, fp32 cuBLAS (TC overhead outweighs benefit)";
  }

  return r;
}

}  // namespace gemm
