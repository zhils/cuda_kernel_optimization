#include "fused/fused_api.h"

#include "fused/fused_gpu.h"

namespace fused {
namespace {

FusedParams FromC(const CkoFusedParams* c) {
  FusedParams p;
  p.B = c->B;
  p.L = c->L;
  p.D = c->D;
  p.H = c->H;
  p.k_size = c->k_size;
  p.dtype = static_cast<common::DType>(c->dtype);
  p.x = c->x;
  p.W_qkv = c->W_qkv;
  p.b_qkv = c->b_qkv;
  p.W_z = c->W_z;
  p.b_z = c->b_z;
  p.K_conv = c->K_conv;
  p.Q = c->Q;
  p.K = c->K;
  p.V = c->V;
  p.impl = static_cast<ImplId>(c->impl);
  return p;
}

}  // namespace

common::Status ValidateFusedParams(const FusedParams& p, bool require_device_ptrs) {
  // 形状与参数校验
  if (p.B <= 0 || p.L <= 0 || p.D <= 0 || p.H <= 0) {
    return common::Status::InvalidArgument("B, L, D, H must be positive");
  }
  if (p.k_size <= 0) {
    return common::Status::InvalidArgument("k_size must be positive");
  }

  // 因果卷积约束校验
  if (p.k_size > p.L) {
    return common::Status::InvalidArgument("k_size cannot exceed L for causal conv");
  }

  // 数据类型校验
  if (p.dtype != common::DType::kFp32 && p.dtype != common::DType::kFp16 &&
      p.dtype != common::DType::kBf16 && p.dtype != common::DType::kInt8 &&
      p.dtype != common::DType::kFp8E4M3 && p.dtype != common::DType::kFp8E5M2) {
    return common::Status::Unsupported("unsupported dtype");
  }

  // 设备指针校验
  if (require_device_ptrs) {
    if (!p.x || !p.W_qkv || !p.b_qkv || !p.W_z || !p.b_z || !p.K_conv || !p.Q || !p.K ||
        !p.V) {
      return common::Status::InvalidArgument("all tensor pointers required for Run");
    }
  }

  return common::Status::Ok();
}

common::Status FusedRun(const FusedParams& p, cudaStream_t stream) {
  // 参数校验
  common::Status st = ValidateFusedParams(p, true);
  if (!st.ok()) return st;

  // 执行 GPU 内核
  st = FusedRunGpu(p, stream);
  if (!st.ok()) return st;

  // 检查异步错误
  const cudaError_t sync_err = cudaGetLastError();
  if (sync_err != cudaSuccess) {
    return common::Status::CudaError(cudaGetErrorString(sync_err));
  }
  return common::Status::Ok();
}

}  // namespace fused

extern "C" {

int cko_fused_conv1d_silu_validate(const CkoFusedParams* params, char* err_buf,
                                   size_t err_len) {
  if (!params) {
    common::CopyStatusMessage(common::Status::InvalidArgument("null params"), err_buf, err_len);
    return CKO_INVALID_ARGUMENT;
  }
  const common::Status st = fused::ValidateFusedParams(fused::FromC(params), false);
  common::CopyStatusMessage(st, err_buf, err_len);
  return common::StatusToC(st.code);
}

int cko_fused_conv1d_silu_run(const CkoFusedParams* params, void* stream) {
  if (!params) return CKO_INVALID_ARGUMENT;
  const common::Status st =
      fused::FusedRun(fused::FromC(params), static_cast<cudaStream_t>(stream));
  return common::StatusToC(st.code);
}

}  // extern "C"
