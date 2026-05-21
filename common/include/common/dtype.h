#pragma once

#include <cstring>

namespace common {

enum class DType : int {
  kFp32 = 0,
  kFp16 = 1,
  kBf16 = 2,
  kFp8E4M3 = 3,
  kFp8E5M2 = 4,
  kInt8 = 5,
  kInt32 = 6,
};

enum class Layout : int {
  kRowMajor = 0,
  kColMajor = 1,
};

inline const char* DTypeName(DType dt) {
  switch (dt) {
    case DType::kFp32: return "fp32";
    case DType::kFp16: return "fp16";
    case DType::kBf16: return "bf16";
    case DType::kFp8E4M3: return "fp8_e4m3";
    case DType::kFp8E5M2: return "fp8_e5m2";
    case DType::kInt8: return "int8";
  }
  return "unknown";
}

inline bool ParseDType(const char* s, DType* out) {
  if (!s || !out) return false;
  if (std::strcmp(s, "fp32") == 0) {
    *out = DType::kFp32;
    return true;
  }
  if (std::strcmp(s, "fp16") == 0) {
    *out = DType::kFp16;
    return true;
  }
  if (std::strcmp(s, "bf16") == 0) {
    *out = DType::kBf16;
    return true;
  }
  if (std::strcmp(s, "fp8_e4m3") == 0) {
    *out = DType::kFp8E4M3;
    return true;
  }
  if (std::strcmp(s, "fp8_e5m2") == 0) {
    *out = DType::kFp8E5M2;
    return true;
  }
  if (std::strcmp(s, "int8") == 0) {
    *out = DType::kInt8;
    return true;
  }
  return false;
}

inline size_t DTypeBytes(DType dt) {
  switch (dt) {
    case DType::kFp32: return 4;
    case DType::kFp16: return 2;
    case DType::kBf16: return 2;
    case DType::kFp8E4M3:
    case DType::kFp8E5M2:
    case DType::kInt8: return 1;
    case DType::kInt32: return 4;
  }
  return 4;
}

}  // namespace common

#ifdef __cplusplus
extern "C" {
#endif

enum CkoDtype {
  CKO_DTYPE_FP32 = 0,
  CKO_DTYPE_FP16 = 1,
  CKO_DTYPE_BF16 = 2,
  CKO_DTYPE_FP8_E4M3 = 3,
  CKO_DTYPE_FP8_E5M2 = 4,
  CKO_DTYPE_INT8 = 5,
  CKO_DTYPE_INT32 = 6,
};

enum CkoLayout {
  CKO_LAYOUT_ROW_MAJOR = 0,
  CKO_LAYOUT_COL_MAJOR = 1,
};

#ifdef __cplusplus
}
#endif
