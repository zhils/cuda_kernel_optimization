#include "fused/fused_ref_cpu.h"

#include <cmath>
#include <vector>

namespace fused {

void FusedConv1dSiLUCore(const float* x, const float* W_qkv, const float* b_qkv,
                         const float* W_z, const float* b_z, const float* K_conv, float* Q,
                         float* K, float* V, int B, int L, int D, int H, int k_size) {
  for (int ib = 0; ib < B; ++ib) {
    for (int t = 0; t < L; ++t) {
      const float* x_bt = x + (ib * L + t) * D;
      for (int h = 0; h < H; ++h) {
        float q_raw = b_qkv[h];
        float k_raw = b_qkv[h + H];
        float v_raw = b_qkv[h + 2 * H];
        for (int d = 0; d < D; ++d) {
          const float xv = x_bt[d];
          q_raw += xv * W_qkv[h * D + d];
          k_raw += xv * W_qkv[(h + H) * D + d];
          v_raw += xv * W_qkv[(h + 2 * H) * D + d];
        }
        Q[(ib * L + t) * H + h] = q_raw;
        K[(ib * L + t) * H + h] = k_raw;
        V[(ib * L + t) * H + h] = v_raw;
      }
    }
  }

  std::vector<float> z_proj(static_cast<size_t>(B) * L * H);
  for (int ib = 0; ib < B; ++ib) {
    for (int t = 0; t < L; ++t) {
      const float* x_bt = x + (ib * L + t) * D;
      for (int h = 0; h < H; ++h) {
        float zp = b_z[h];
        for (int d = 0; d < D; ++d) {
          zp += x_bt[d] * W_z[h * D + d];
        }
        z_proj[(ib * L + t) * H + h] = zp;
      }
    }
  }

  for (int ib = 0; ib < B; ++ib) {
    for (int t = 0; t < L; ++t) {
      for (int h = 0; h < H; ++h) {
        float z_conv = 0.f;
        for (int i = 0; i < k_size; ++i) {
          const int ti = t - i;
          if (ti < 0) continue;
          z_conv += z_proj[(ib * L + ti) * H + h] * K_conv[i * H + h];
        }
        const float sigmoid = 1.f / (1.f + std::exp(-z_conv));
        V[(ib * L + t) * H + h] *= (z_conv * sigmoid);
      }
    }
  }
}

float DefaultAbsTolerance() { return 1e-2f; }

common::Status FusedReferenceHost(const FusedParams& p) {
  common::Status st = ValidateFusedParams(p, false);
  if (!st.ok()) return st;
  if (!p.x || !p.W_qkv || !p.b_qkv || !p.W_z || !p.b_z || !p.K_conv || !p.Q || !p.K || !p.V) {
    return common::Status::InvalidArgument("all host tensor pointers required");
  }
  if (p.dtype != common::DType::kFp32) {
    return common::Status::Unsupported("FusedReferenceHost supports fp32 only");
  }

  FusedConv1dSiLUCore(reinterpret_cast<const float*>(p.x),
                      reinterpret_cast<const float*>(p.W_qkv),
                      reinterpret_cast<const float*>(p.b_qkv),
                      reinterpret_cast<const float*>(p.W_z),
                      reinterpret_cast<const float*>(p.b_z),
                      reinterpret_cast<const float*>(p.K_conv),
                      reinterpret_cast<float*>(p.Q), reinterpret_cast<float*>(p.K),
                      reinterpret_cast<float*>(p.V), p.B, p.L, p.D, p.H, p.k_size);
  return common::Status::Ok();
}

}  // namespace fused
