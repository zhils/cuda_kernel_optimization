#pragma once

#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "rmsnorm/rmsnorm_dtype.h"

namespace rmsnorm {

inline float QuantMax(ActivationDtype dt) {
  switch (dt) {
    case ActivationDtype::kInt8:
      return 127.f;
    case ActivationDtype::kFp8E4M3:
      return 448.f;
    case ActivationDtype::kFp8E5M2:
      return 57344.f;
    default:
      return 1.f;
  }
}

inline float RowMaxAbs(const float* row, int cols) {
  float m = 0.f;
  for (int c = 0; c < cols; ++c) m = fmaxf(m, fabsf(row[c]));
  return m;
}

inline float RowScaleFromMax(float max_abs, float quant_max) {
  if (max_abs < 1e-8f) return 1.f;
  return max_abs / quant_max;
}

inline int8_t QuantizeInt8(float v, float scale) {
  const float q = v / scale;
  const float clamped = fminf(fmaxf(q, -128.f), 127.f);
  return static_cast<int8_t>(lrintf(clamped));
}

inline float DequantizeInt8(int8_t q, float scale) { return static_cast<float>(q) * scale; }

inline __nv_fp8_e4m3 QuantizeFp8E4M3(float v, float scale) {
  return __nv_fp8_e4m3(v / scale);
}

inline float DequantizeFp8E4M3(__nv_fp8_e4m3 q, float scale) {
  return static_cast<float>(q) * scale;
}

inline __nv_fp8_e5m2 QuantizeFp8E5M2(float v, float scale) {
  return __nv_fp8_e5m2(v / scale);
}

inline float DequantizeFp8E5M2(__nv_fp8_e5m2 q, float scale) {
  return static_cast<float>(q) * scale;
}

inline void QuantizeActivationHost(ActivationDtype dt, const std::vector<float>& x_fp32,
                                   int rows, int cols, std::vector<uint8_t>& storage,
                                   std::vector<float>& scales) {
  const float qmax = QuantMax(dt);
  scales.assign(rows, 1.f);
  storage.assign(static_cast<size_t>(rows) * cols, 0);

  for (int r = 0; r < rows; ++r) {
    const float* row = x_fp32.data() + static_cast<size_t>(r) * cols;
    const float scale = RowScaleFromMax(RowMaxAbs(row, cols), qmax);
    scales[r] = scale;
    for (int c = 0; c < cols; ++c) {
      const size_t idx = static_cast<size_t>(r) * cols + c;
      const float v = row[c];
      switch (dt) {
        case ActivationDtype::kInt8:
          storage[idx] = static_cast<uint8_t>(QuantizeInt8(v, scale));
          break;
        case ActivationDtype::kFp8E4M3: {
          const __nv_fp8_e4m3 q = QuantizeFp8E4M3(v, scale);
          storage[idx] = reinterpret_cast<const uint8_t&>(q);
          break;
        }
        case ActivationDtype::kFp8E5M2: {
          const __nv_fp8_e5m2 q = QuantizeFp8E5M2(v, scale);
          storage[idx] = reinterpret_cast<const uint8_t&>(q);
          break;
        }
        default:
          break;
      }
    }
  }
}

inline float DequantizeActivationHost(ActivationDtype dt, uint8_t code, float scale) {
  switch (dt) {
    case ActivationDtype::kInt8:
      return DequantizeInt8(static_cast<int8_t>(code), scale);
    case ActivationDtype::kFp8E4M3: {
      __nv_fp8_e4m3 q;
      reinterpret_cast<uint8_t&>(q) = code;
      return DequantizeFp8E4M3(q, scale);
    }
    case ActivationDtype::kFp8E5M2: {
      __nv_fp8_e5m2 q;
      reinterpret_cast<uint8_t&>(q) = code;
      return DequantizeFp8E5M2(q, scale);
    }
    default:
      return 0.f;
  }
}

inline void RMSNormQuantizedCPU(ActivationDtype dt, const std::vector<uint8_t>& x_q,
                                const std::vector<float>& x_scale,
                                const std::vector<float>& w_fp32, std::vector<uint8_t>& y_q,
                                std::vector<float>& y_scale, int rows, int cols, float eps) {
  const float qmax = QuantMax(dt);
  y_q.assign(x_q.size(), 0);
  y_scale.assign(rows, 1.f);

  std::vector<float> y_fp32(static_cast<size_t>(rows) * cols);
  for (int r = 0; r < rows; ++r) {
    float sq_sum = 0.f;
    for (int c = 0; c < cols; ++c) {
      const size_t idx = static_cast<size_t>(r) * cols + c;
      const float val =
          DequantizeActivationHost(dt, x_q[idx], x_scale[r]);
      sq_sum += val * val;
    }
    const float rms = 1.f / sqrtf(sq_sum / cols + eps);
    float* y_row = y_fp32.data() + static_cast<size_t>(r) * cols;
    for (int c = 0; c < cols; ++c) {
      const size_t idx = static_cast<size_t>(r) * cols + c;
      const float val =
          DequantizeActivationHost(dt, x_q[idx], x_scale[r]);
      y_row[c] = val * rms * w_fp32[c];
    }
    const float out_scale = RowScaleFromMax(RowMaxAbs(y_row, cols), qmax);
    y_scale[r] = out_scale;
    for (int c = 0; c < cols; ++c) {
      const size_t idx = static_cast<size_t>(r) * cols + c;
      const float v = y_row[c];
      switch (dt) {
        case ActivationDtype::kInt8:
          y_q[idx] = static_cast<uint8_t>(QuantizeInt8(v, out_scale));
          break;
        case ActivationDtype::kFp8E4M3: {
          const __nv_fp8_e4m3 q = QuantizeFp8E4M3(v, out_scale);
          y_q[idx] = reinterpret_cast<const uint8_t&>(q);
          break;
        }
        case ActivationDtype::kFp8E5M2: {
          const __nv_fp8_e5m2 q = QuantizeFp8E5M2(v, out_scale);
          y_q[idx] = reinterpret_cast<const uint8_t&>(q);
          break;
        }
        default:
          break;
      }
    }
  }
}

inline std::vector<float> DequantizeMatrixHost(ActivationDtype dt,
                                               const std::vector<uint8_t>& q,
                                               const std::vector<float>& scales, int rows,
                                               int cols) {
  std::vector<float> out(static_cast<size_t>(rows) * cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const size_t idx = static_cast<size_t>(r) * cols + c;
      out[idx] = DequantizeActivationHost(dt, q[idx], scales[r]);
    }
  }
  return out;
}

}  // namespace rmsnorm
