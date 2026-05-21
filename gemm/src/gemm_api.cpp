#include "gemm/gemm_api.h"

#include "gemm/gemm_gpu.h"

namespace gemm {
namespace {

common::DType FromCDtype(int v) { return static_cast<common::DType>(v); }

bool IsFp16Inputs(const GemmParams& p) {
  return p.dtype_a == common::DType::kFp16 && p.dtype_b == common::DType::kFp16;
}

bool IsBf16Inputs(const GemmParams& p) {
  return p.dtype_a == common::DType::kBf16 && p.dtype_b == common::DType::kBf16;
}

bool IsInt8Inputs(const GemmParams& p) {
  return p.dtype_a == common::DType::kInt8 && p.dtype_b == common::DType::kInt8;
}

bool IsFp8Inputs(const GemmParams& p) {
  return (p.dtype_a == common::DType::kFp8E4M3 || p.dtype_a == common::DType::kFp8E5M2) &&
         p.dtype_a == p.dtype_b;
}

ImplId FallbackImpl(const GemmParams& p) {
  if (IsFp16Inputs(p)) return ImplId::kCublasFp16;
  if (IsBf16Inputs(p)) return ImplId::kCublasBf16;
  if (IsInt8Inputs(p)) return ImplId::kCublasInt8;
  if (IsFp8Inputs(p)) return ImplId::kCublasFp8;
  return ImplId::kCublas;
}

GemmParams FromC(const CkoGemmParams* c) {
  GemmParams p;
  p.M = c->M;
  p.N = c->N;
  p.K = c->K;
  p.dtype_a = FromCDtype(c->dtype_a);
  p.dtype_b = FromCDtype(c->dtype_b);
  p.dtype_c = FromCDtype(c->dtype_c);
  p.layout = static_cast<common::Layout>(c->layout);
  p.alpha = c->alpha;
  p.beta = c->beta;
  p.A = c->A;
  p.B = c->B;
  p.C = c->C;
  p.lda = c->lda;
  p.ldb = c->ldb;
  p.ldc = c->ldc;
  p.impl = static_cast<ImplId>(c->impl);
  p.alignment_policy = static_cast<AlignmentPolicy>(c->alignment_policy);
  return p;
}

}  // namespace

common::Status ValidateGemmParams(const GemmParams& p, bool require_device_ptrs) {
  // 形状校验
  if (p.M <= 0 || p.N <= 0 || p.K <= 0) {
    return common::Status::InvalidArgument("M, N, K must be positive");
  }

  // 主维度校验
  if (p.lda <= 0 || p.ldb <= 0 || p.ldc <= 0) {
    return common::Status::InvalidArgument("lda, ldb, ldc must be positive");
  }

  // 布局校验
  if (p.layout != common::Layout::kRowMajor) {
    return common::Status::Unsupported("only row_major layout in Phase 0");
  }

  // 主维度与形状一致性校验
  const int min_ld_a = p.K;
  const int min_ld_b = p.N;
  const int min_ld_c = p.N;
  if (p.lda < min_ld_a || p.ldb < min_ld_b || p.ldc < min_ld_c) {
    return common::Status::InvalidArgument("leading dimensions too small for shape");
  }

  // 设备指针校验
  if (require_device_ptrs && (p.A == nullptr || p.B == nullptr || p.C == nullptr)) {
    return common::Status::InvalidArgument("A, B, C device pointers required for Run");
  }

  return common::Status::Ok();
}

bool IsAlignedForTiledKernel(const GemmParams& p, const GemmTileRequirements& req) {
  return (p.M % req.block_m == 0) && (p.N % req.block_n == 0) && (p.K % req.tile_k == 0);
}

ImplId ResolveImpl(const GemmParams& p, common::Status* alignment_note) {
  GemmTileRequirements req;
  ImplId chosen = p.impl;

  // 自动选择实现
  if (chosen == ImplId::kAuto) {
    if (IsFp8Inputs(p)) chosen = ImplId::kCublasFp8;
    else if (IsInt8Inputs(p)) chosen = ImplId::kCublasInt8;
    else if (IsBf16Inputs(p)) chosen = ImplId::kCublasBf16;
    else if (IsFp16Inputs(p)) chosen = ImplId::kFp16;
    else chosen = ImplId::kV3;
  }

  // 对齐校验与回退
  if (chosen == ImplId::kV3 || chosen == ImplId::kV4 || chosen == ImplId::kFp16) {
    const bool aligned =
        IsAlignedForImpl(chosen, p) || IsAlignedForTiledKernel(p, req);
    if (!aligned) {
      if (p.alignment_policy == AlignmentPolicy::kStrict) {
        if (alignment_note) {
          *alignment_note = common::Status::InvalidArgument(
              "M/N/K not aligned for tiled kernel; use cublas or change policy");
        }
        return chosen;
      }
      if (p.alignment_policy == AlignmentPolicy::kSkip) {
        if (alignment_note) {
          *alignment_note = common::Status::Skip("unaligned for tiled GEMM");
        }
        return chosen;
      }
      if (alignment_note) {
        *alignment_note = common::Status::Ok();
        alignment_note->message = "fallback: unaligned shape → cuBLAS path";
      }
      return FallbackImpl(p);
    }
  }

  if (alignment_note) {
    *alignment_note = common::Status::Ok();
  }
  return chosen;
}

common::Status GemmRun(const GemmParams& p, cudaStream_t stream) {
  // 参数校验
  common::Status st = ValidateGemmParams(p, true);
  if (!st.ok()) return st;

  // 解析实现
  common::Status note;
  GemmParams resolved = p;
  resolved.impl = ResolveImpl(p, &note);
  if (note.code == common::StatusCode::kInvalidArgument ||
      note.code == common::StatusCode::kSkip) {
    return note;
  }

  // 执行 GPU 内核
  st = GemmRunGpu(resolved, stream);
  if (!st.ok()) return st;

  // 检查异步错误
  const cudaError_t sync_err = cudaGetLastError();
  if (sync_err != cudaSuccess) {
    return common::Status::CudaError(cudaGetErrorString(sync_err));
  }
  return common::Status::Ok();
}

}  // namespace gemm

extern "C" {

int cko_gemm_validate(const CkoGemmParams* params, char* err_buf, size_t err_len) {
  if (!params) {
    common::CopyStatusMessage(common::Status::InvalidArgument("null params"), err_buf, err_len);
    return CKO_INVALID_ARGUMENT;
  }
  gemm::GemmParams p = gemm::FromC(params);
  const common::Status st = gemm::ValidateGemmParams(p, false);
  common::CopyStatusMessage(st, err_buf, err_len);
  return common::StatusToC(st.code);
}

int cko_gemm_run(const CkoGemmParams* params, void* stream) {
  if (!params) {
    return CKO_INVALID_ARGUMENT;
  }
  const common::Status st = gemm::GemmRun(gemm::FromC(params),
                                          static_cast<cudaStream_t>(stream));
  return common::StatusToC(st.code);
}

}  // extern "C"
