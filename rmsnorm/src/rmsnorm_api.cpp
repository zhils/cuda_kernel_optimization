#include "rmsnorm/rmsnorm_api.h"

#include "rmsnorm/rmsnorm_gpu.h"

namespace rmsnorm {
namespace {

RmsNormParams FromC(const CkoRmsNormParams* c) {
  RmsNormParams p;
  p.rows = c->rows;
  p.cols = c->cols;
  p.act_dtype = static_cast<common::DType>(c->act_dtype);
  p.weight_dtype = static_cast<common::DType>(c->weight_dtype);
  p.eps = c->eps > 0.f ? c->eps : 1e-5f;
  p.input = c->input;
  p.weight = c->weight;
  p.output = c->output;
  p.input_scale = c->input_scale;
  p.output_scale = c->output_scale;
  p.impl = static_cast<ImplId>(c->impl);
  return p;
}

bool IsQuantizedAct(common::DType dt) {
  return dt == common::DType::kFp8E4M3 || dt == common::DType::kFp8E5M2 ||
         dt == common::DType::kInt8;
}

}  // namespace

common::Status ValidateRmsNormParams(const RmsNormParams& p, bool require_device_ptrs) {
  // 形状与参数校验
  if (p.rows <= 0 || p.cols <= 0) {
    return common::Status::InvalidArgument("rows and cols must be positive");
  }
  if (p.eps <= 0.f) {
    return common::Status::InvalidArgument("eps must be positive");
  }

  // 权重类型校验
  if (p.weight_dtype != common::DType::kFp32 && p.weight_dtype != common::DType::kFp16 &&
      p.weight_dtype != common::DType::kBf16) {
    return common::Status::Unsupported("weight must be fp32, fp16, or bf16");
  }

  // 量化激活 scale 校验
  if (IsQuantizedAct(p.act_dtype)) {
    if (require_device_ptrs && (p.input_scale == nullptr || p.output_scale == nullptr)) {
      return common::Status::InvalidArgument("quantized activation requires per-row scales");
    }
  }

  // 设备指针校验
  if (require_device_ptrs &&
      (p.input == nullptr || p.weight == nullptr || p.output == nullptr)) {
    return common::Status::InvalidArgument("input, weight, output pointers required for Run");
  }

  return common::Status::Ok();
}

common::Status RmsNormRun(const RmsNormParams& p, cudaStream_t stream) {
  // 参数校验
  common::Status st = ValidateRmsNormParams(p, true);
  if (!st.ok()) return st;

  // 执行 GPU 内核
  st = RmsNormRunGpu(p, stream);
  if (!st.ok()) return st;

  // 检查异步错误
  const cudaError_t sync_err = cudaGetLastError();
  if (sync_err != cudaSuccess) {
    return common::Status::CudaError(cudaGetErrorString(sync_err));
  }
  return common::Status::Ok();
}

}  // namespace rmsnorm

extern "C" {

int cko_rmsnorm_validate(const CkoRmsNormParams* params, char* err_buf, size_t err_len) {
  if (!params) {
    common::CopyStatusMessage(common::Status::InvalidArgument("null params"), err_buf, err_len);
    return CKO_INVALID_ARGUMENT;
  }
  const common::Status st = rmsnorm::ValidateRmsNormParams(rmsnorm::FromC(params), false);
  common::CopyStatusMessage(st, err_buf, err_len);
  return common::StatusToC(st.code);
}

int cko_rmsnorm_run(const CkoRmsNormParams* params, void* stream) {
  if (!params) return CKO_INVALID_ARGUMENT;
  const common::Status st =
      rmsnorm::RmsNormRun(rmsnorm::FromC(params), static_cast<cudaStream_t>(stream));
  return common::StatusToC(st.code);
}

}  // extern "C"
