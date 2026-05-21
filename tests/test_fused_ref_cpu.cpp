#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "common/benchmark.h"
#include "common/test_case.h"
#include "fused/fused_api.h"
#include "fused/fused_ref_cpu.h"

#ifndef CKO_SOURCE_DIR
#define CKO_SOURCE_DIR "."
#endif

namespace {

void FillDeterministic(std::vector<float>& x, int n, float scale, float bias) {
  for (int i = 0; i < n; ++i) {
    x[static_cast<size_t>(i)] = scale * static_cast<float>(i + 1) + bias;
  }
}

fused::FusedParams MakeParams(int B, int L, int D, int H, int k_size, std::vector<float>& x,
                              std::vector<float>& W_qkv, std::vector<float>& b_qkv,
                              std::vector<float>& W_z, std::vector<float>& b_z,
                              std::vector<float>& K_conv, std::vector<float>& Q,
                              std::vector<float>& K, std::vector<float>& V) {
  fused::FusedParams p;
  p.B = B;
  p.L = L;
  p.D = D;
  p.H = H;
  p.k_size = k_size;
  p.dtype = common::DType::kFp32;
  p.x = x.data();
  p.W_qkv = W_qkv.data();
  p.b_qkv = b_qkv.data();
  p.W_z = W_z.data();
  p.b_z = b_z.data();
  p.K_conv = K_conv.data();
  p.Q = Q.data();
  p.K = K.data();
  p.V = V.data();
  return p;
}

}  // namespace

TEST(FusedRefCpuTest, DeterministicFinite) {
  constexpr int B = 2;
  constexpr int L = 8;
  constexpr int D = 16;
  constexpr int H = 8;
  constexpr int k_size = 4;

  std::vector<float> x(static_cast<size_t>(B) * L * D);
  std::vector<float> W_qkv(static_cast<size_t>(3) * H * D);
  std::vector<float> b_qkv(static_cast<size_t>(3) * H);
  std::vector<float> W_z(static_cast<size_t>(H) * D);
  std::vector<float> b_z(H);
  std::vector<float> K_conv(static_cast<size_t>(k_size) * H);
  std::vector<float> Q(static_cast<size_t>(B) * L * H);
  std::vector<float> K(static_cast<size_t>(B) * L * H);
  std::vector<float> V(static_cast<size_t>(B) * L * H);

  FillDeterministic(x, static_cast<int>(x.size()), 0.01f, -0.1f);
  FillDeterministic(W_qkv, static_cast<int>(W_qkv.size()), 0.002f, 0.f);
  FillDeterministic(b_qkv, static_cast<int>(b_qkv.size()), 0.01f, 0.f);
  FillDeterministic(W_z, static_cast<int>(W_z.size()), 0.003f, 0.f);
  FillDeterministic(b_z, H, 0.02f, 0.f);
  FillDeterministic(K_conv, static_cast<int>(K_conv.size()), 0.05f, 0.f);

  const auto p = MakeParams(B, L, D, H, k_size, x, W_qkv, b_qkv, W_z, b_z, K_conv, Q, K, V);
  ASSERT_TRUE(fused::FusedReferenceHost(p).ok());

  for (size_t i = 0; i < Q.size(); ++i) {
    EXPECT_TRUE(std::isfinite(Q[i]));
    EXPECT_TRUE(std::isfinite(K[i]));
    EXPECT_TRUE(std::isfinite(V[i]));
  }
}

TEST(FusedRefCpuTest, CatalogCasesReferenceOk) {
  const auto cat = common::LoadTestCaseCatalog(
      std::string(CKO_SOURCE_DIR) + "/configs/test_cases/fused_conv1d_silu.json");
  for (const auto& c : cat.cases) {
    if (c.expect != common::ExpectStatus::kPass) continue;
    if (c.id == "main_demo") continue;

    const int B = c.fused.B;
    const int L = c.fused.L;
    const int D = c.fused.D;
    const int H = c.fused.H;
    const int k_size = c.fused.k_size;

    std::vector<float> x(static_cast<size_t>(B) * L * D);
    std::vector<float> W_qkv(static_cast<size_t>(3) * H * D);
    std::vector<float> b_qkv(static_cast<size_t>(3) * H);
    std::vector<float> W_z(static_cast<size_t>(H) * D);
    std::vector<float> b_z(H);
    std::vector<float> K_conv(static_cast<size_t>(k_size) * H);
    std::vector<float> Q(static_cast<size_t>(B) * L * H);
    std::vector<float> K(static_cast<size_t>(B) * L * H);
    std::vector<float> V(static_cast<size_t>(B) * L * H);

    FillDeterministic(x, static_cast<int>(x.size()), 0.008f, 0.f);
    FillDeterministic(W_qkv, static_cast<int>(W_qkv.size()), 0.001f, 0.f);
    FillDeterministic(b_qkv, static_cast<int>(b_qkv.size()), 0.005f, 0.f);
    FillDeterministic(W_z, static_cast<int>(W_z.size()), 0.0015f, 0.f);
    FillDeterministic(b_z, H, 0.004f, 0.f);
    FillDeterministic(K_conv, static_cast<int>(K_conv.size()), 0.02f, 0.f);

    const auto p = MakeParams(B, L, D, H, k_size, x, W_qkv, b_qkv, W_z, b_z, K_conv, Q, K, V);
    const common::Status st = fused::FusedReferenceHost(p);
    EXPECT_TRUE(st.ok()) << c.id << " msg=" << st.message;
  }
}
