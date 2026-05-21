#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace rmsnorm {

enum class ActivationDtype {
  kFp32,
  kFp16,
  kBf16,
  kFp8E4M3,  // Phase 2: dynamic per-row scale
  kFp8E5M2,  // Phase 2: dynamic per-row scale
  kInt8,     // Phase 2: dynamic per-row scale
};

enum class WeightDtype {
  kFp32,
  kFp16,
  kBf16,
};

inline bool ParseActivationDtype(const char* s, ActivationDtype* out) {
  if (!s || !out) return false;
  if (std::strcmp(s, "fp32") == 0) {
    *out = ActivationDtype::kFp32;
    return true;
  }
  if (std::strcmp(s, "fp16") == 0) {
    *out = ActivationDtype::kFp16;
    return true;
  }
  if (std::strcmp(s, "bf16") == 0) {
    *out = ActivationDtype::kBf16;
    return true;
  }
  if (std::strcmp(s, "fp8_e4m3") == 0) {
    *out = ActivationDtype::kFp8E4M3;
    return true;
  }
  if (std::strcmp(s, "fp8_e5m2") == 0) {
    *out = ActivationDtype::kFp8E5M2;
    return true;
  }
  if (std::strcmp(s, "int8") == 0) {
    *out = ActivationDtype::kInt8;
    return true;
  }
  return false;
}

inline bool ParseWeightDtype(const char* s, WeightDtype* out) {
  if (!s || !out) return false;
  if (std::strcmp(s, "fp32") == 0) {
    *out = WeightDtype::kFp32;
    return true;
  }
  if (std::strcmp(s, "fp16") == 0) {
    *out = WeightDtype::kFp16;
    return true;
  }
  if (std::strcmp(s, "bf16") == 0) {
    *out = WeightDtype::kBf16;
    return true;
  }
  return false;
}

inline const char* ActivationDtypeName(ActivationDtype dt) {
  switch (dt) {
    case ActivationDtype::kFp32: return "fp32";
    case ActivationDtype::kFp16: return "fp16";
    case ActivationDtype::kBf16: return "bf16";
    case ActivationDtype::kFp8E4M3: return "fp8_e4m3";
    case ActivationDtype::kFp8E5M2: return "fp8_e5m2";
    case ActivationDtype::kInt8: return "int8";
  }
  return "unknown";
}

inline const char* WeightDtypeName(WeightDtype dt) {
  switch (dt) {
    case WeightDtype::kFp32: return "fp32";
    case WeightDtype::kFp16: return "fp16";
    case WeightDtype::kBf16: return "bf16";
  }
  return "unknown";
}

inline size_t ActivationElemBytes(ActivationDtype dt) {
  switch (dt) {
    case ActivationDtype::kFp32: return 4;
    case ActivationDtype::kFp16: return 2;
    case ActivationDtype::kBf16: return 2;
    case ActivationDtype::kFp8E4M3:
    case ActivationDtype::kFp8E5M2: return 1;
    case ActivationDtype::kInt8: return 1;
  }
  return 4;
}

inline size_t WeightElemBytes(WeightDtype dt) {
  switch (dt) {
    case WeightDtype::kFp32: return 4;
    case WeightDtype::kFp16: return 2;
    case WeightDtype::kBf16: return 2;
  }
  return 4;
}

inline bool IsP0Supported(ActivationDtype act) {
  return act == ActivationDtype::kFp32 || act == ActivationDtype::kFp16 ||
         act == ActivationDtype::kBf16;
}

inline bool IsQuantizedActivation(ActivationDtype act) {
  return act == ActivationDtype::kFp8E4M3 || act == ActivationDtype::kFp8E5M2 ||
         act == ActivationDtype::kInt8;
}

struct LaunchConfig {
  ActivationDtype act_dtype = ActivationDtype::kFp32;
  WeightDtype weight_dtype = WeightDtype::kFp32;
};

inline LaunchConfig ParseArgs(int argc, char** argv) {
  LaunchConfig cfg;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) {
      if (!ParseActivationDtype(argv[++i], &cfg.act_dtype)) {
        std::cerr << "Unknown --dtype: " << argv[i] << "\n";
        std::exit(EXIT_FAILURE);
      }
    } else if (std::strcmp(argv[i], "--weight-dtype") == 0 && i + 1 < argc) {
      if (!ParseWeightDtype(argv[++i], &cfg.weight_dtype)) {
        std::cerr << "Unknown --weight-dtype: " << argv[i] << "\n";
        std::exit(EXIT_FAILURE);
      }
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      std::cout << "Usage: rmsnorm_v3 [--dtype fp32|fp16|bf16|fp8_e4m3|fp8_e5m2|int8]\n"
                << "                [--weight-dtype fp32|fp16|bf16]\n"
                << "Quantized activations (fp8/int8) use dynamic per-row input/output scale.\n"
                << "Weight is always fp32/fp16/bf16; only activation may be quantized.\n";
      std::exit(EXIT_SUCCESS);
    }
  }
  return cfg;
}

inline void ValidateConfig(const LaunchConfig& cfg) {
  if (!IsP0Supported(cfg.act_dtype) && !IsQuantizedActivation(cfg.act_dtype)) {
    std::cerr << "Unsupported activation dtype.\n";
    std::exit(EXIT_FAILURE);
  }
}

inline __device__ float LoadWeightToFloat(const void* weight, WeightDtype dt, int c) {
  switch (dt) {
    case WeightDtype::kFp32:
      return reinterpret_cast<const float*>(weight)[c];
    case WeightDtype::kFp16:
      return __half2float(reinterpret_cast<const __half*>(weight)[c]);
    case WeightDtype::kBf16:
      return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(weight)[c]);
  }
  return 0.f;
}

}  // namespace rmsnorm
