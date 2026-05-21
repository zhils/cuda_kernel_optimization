#include "rmsnorm/rmsnorm_ref_cpu.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstring>

#include "rmsnorm/rmsnorm_dtype.h"
#include "rmsnorm/rmsnorm_quant.h"

namespace rmsnorm {
namespace {

ActivationDtype ToActivationDtype(common::DType dt) {
  switch (dt) {
    case common::DType::kFp32: return ActivationDtype::kFp32;
    case common::DType::kFp16: return ActivationDtype::kFp16;
    case common::DType::kBf16: return ActivationDtype::kBf16;
    case common::DType::kFp8E4M3: return ActivationDtype::kFp8E4M3;
    case common::DType::kFp8E5M2: return ActivationDtype::kFp8E5M2;
    case common::DType::kInt8: return ActivationDtype::kInt8;
  }
  return ActivationDtype::kFp32;
}

WeightDtype ToWeightDtype(common::DType dt) {
  switch (dt) {
    case common::DType::kFp32: return WeightDtype::kFp32;
    case common::DType::kFp16: return WeightDtype::kFp16;
    case common::DType::kBf16: return WeightDtype::kBf16;
    default: return WeightDtype::kFp32;
  }
}

float LoadWeightHost(const void* weight, WeightDtype dt, int c) {
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

float LoadActHost(const void* input, ActivationDtype dt, int idx) {
  switch (dt) {
    case ActivationDtype::kFp32:
      return reinterpret_cast<const float*>(input)[idx];
    case ActivationDtype::kFp16:
      return __half2float(reinterpret_cast<const __half*>(input)[idx]);
    case ActivationDtype::kBf16:
      return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(input)[idx]);
    default:
      return 0.f;
  }
}

void StoreActHost(void* output, ActivationDtype dt, int idx, float v) {
  switch (dt) {
    case ActivationDtype::kFp32:
      reinterpret_cast<float*>(output)[idx] = v;
      break;
    case ActivationDtype::kFp16:
      reinterpret_cast<__half*>(output)[idx] = __float2half(v);
      break;
    case ActivationDtype::kBf16:
      reinterpret_cast<__nv_bfloat16*>(output)[idx] = __float2bfloat16(v);
      break;
    default:
      break;
  }
}

common::Status ReferenceDense(const RmsNormParams& p) {
  const ActivationDtype act = ToActivationDtype(p.act_dtype);
  const WeightDtype wdt = ToWeightDtype(p.weight_dtype);
  const int n = p.rows * p.cols;

  std::vector<float> x_fp32(static_cast<size_t>(n));
  std::vector<float> w_fp32(static_cast<size_t>(p.cols));
  for (int i = 0; i < n; ++i) {
    x_fp32[i] = LoadActHost(p.input, act, i);
  }
  for (int c = 0; c < p.cols; ++c) {
    w_fp32[c] = LoadWeightHost(p.weight, wdt, c);
  }

  std::vector<float> y_fp32(static_cast<size_t>(n));
  RmsNormFp32Core(x_fp32.data(), w_fp32.data(), y_fp32.data(), p.rows, p.cols, p.eps);

  for (int i = 0; i < n; ++i) {
    StoreActHost(p.output, act, i, y_fp32[i]);
  }
  return common::Status::Ok();
}

common::Status ReferenceQuantized(const RmsNormParams& p) {
  const ActivationDtype act = ToActivationDtype(p.act_dtype);
  const auto* x_q = reinterpret_cast<const uint8_t*>(p.input);
  auto* y_q = reinterpret_cast<uint8_t*>(p.output);

  std::vector<float> w_fp32(static_cast<size_t>(p.cols));
  const WeightDtype wdt = ToWeightDtype(p.weight_dtype);
  for (int c = 0; c < p.cols; ++c) {
    w_fp32[c] = LoadWeightHost(p.weight, wdt, c);
  }

  std::vector<uint8_t> x_q_copy(x_q, x_q + static_cast<size_t>(p.rows) * p.cols);
  std::vector<float> x_scale(p.input_scale, p.input_scale + p.rows);
  std::vector<uint8_t> y_q_out;
  std::vector<float> y_scale;
  RMSNormQuantizedCPU(act, x_q_copy, x_scale, w_fp32, y_q_out, y_scale, p.rows, p.cols, p.eps);

  std::memcpy(y_q, y_q_out.data(), y_q_out.size());
  std::memcpy(p.output_scale, y_scale.data(), static_cast<size_t>(p.rows) * sizeof(float));
  return common::Status::Ok();
}

}  // namespace

void RmsNormFp32Core(const float* x, const float* weight, float* y, int rows, int cols,
                     float eps) {
  for (int r = 0; r < rows; ++r) {
    float sq_sum = 0.f;
    for (int c = 0; c < cols; ++c) {
      const float val = x[static_cast<size_t>(r) * cols + c];
      sq_sum += val * val;
    }
    const float rms = 1.f / std::sqrt(sq_sum / static_cast<float>(cols) + eps);
    for (int c = 0; c < cols; ++c) {
      const size_t idx = static_cast<size_t>(r) * cols + c;
      y[idx] = x[idx] * rms * weight[c];
    }
  }
}

float DefaultAbsTolerance(common::DType act_dtype) {
  switch (act_dtype) {
    case common::DType::kFp32: return 1e-4f;
    case common::DType::kFp16:
    case common::DType::kBf16: return 2e-2f;
    case common::DType::kInt8:
    case common::DType::kFp8E4M3:
    case common::DType::kFp8E5M2: return 2e-2f;
  }
  return 1e-2f;
}

common::Status DequantizeOutputToFp32(common::DType act_dtype, const void* output,
                                      const float* output_scale, int rows, int cols,
                                      std::vector<float>& out_fp32) {
  const ActivationDtype act = ToActivationDtype(act_dtype);
  const int n = rows * cols;
  out_fp32.assign(static_cast<size_t>(n), 0.f);

  if (IsQuantizedActivation(act)) {
    if (!output_scale) {
      return common::Status::InvalidArgument("output_scale required for quantized activation");
    }
    const auto* q = reinterpret_cast<const uint8_t*>(output);
    std::vector<uint8_t> qv(q, q + n);
    std::vector<float> scales(output_scale, output_scale + rows);
    out_fp32 = DequantizeMatrixHost(act, qv, scales, rows, cols);
    return common::Status::Ok();
  }

  for (int i = 0; i < n; ++i) {
    out_fp32[i] = LoadActHost(output, act, i);
  }
  return common::Status::Ok();
}

common::Status RmsNormReferenceHost(const RmsNormParams& p) {
  common::Status st = ValidateRmsNormParams(p, false);
  if (!st.ok()) return st;
  if (p.input == nullptr || p.weight == nullptr || p.output == nullptr) {
    return common::Status::InvalidArgument("input, weight, output host pointers required");
  }

  const ActivationDtype act = ToActivationDtype(p.act_dtype);
  if (IsQuantizedActivation(act)) {
    if (p.input_scale == nullptr || p.output_scale == nullptr) {
      return common::Status::InvalidArgument("quantized activation requires input/output scales");
    }
    return ReferenceQuantized(p);
  }
  return ReferenceDense(p);
}

}  // namespace rmsnorm
