#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "common/benchmark.h"
#include "common/test_case.h"
#include "gemm/gemm_api.h"
#include "gemm/gemm_ref_cpu.h"

#ifndef CKO_SOURCE_DIR
#define CKO_SOURCE_DIR "."
#endif

namespace {

void FillDeterministic(std::vector<float>& m, int rows, int cols, float scale) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      m[static_cast<size_t>(r) * cols + c] =
          scale * (0.01f * static_cast<float>(r + 1) + 0.001f * static_cast<float>(c + 1));
    }
  }
}

gemm::GemmParams MakeFp32Params(int M, int N, int K, std::vector<float>& A, std::vector<float>& B,
                                std::vector<float>& C) {
  gemm::GemmParams p;
  p.M = M;
  p.N = N;
  p.K = K;
  p.lda = K;
  p.ldb = N;
  p.ldc = N;
  p.A = A.data();
  p.B = B.data();
  p.C = C.data();
  return p;
}

bool SkipSlowCpuCatalogCase(const common::TestCaseEntry& c) {
  if (c.id == "main_4096" || c.id == "unaligned_1000") return true;
  const int64_t flops =
      static_cast<int64_t>(c.gemm.M) * c.gemm.N * c.gemm.K;
  return flops > 128LL * 1024 * 1024;
}

}  // namespace

TEST(GemmRefCpuTest, Fp32KnownValues) {
  constexpr int M = 2, N = 2, K = 2;
  std::vector<float> A = {1.f, 2.f, 3.f, 4.f};
  std::vector<float> B = {5.f, 6.f, 7.f, 8.f};
  std::vector<float> C(M * N, 0.f);
  std::vector<float> C_core(M * N, 0.f);

  const auto p = MakeFp32Params(M, N, K, A, B, C);
  ASSERT_TRUE(gemm::GemmReferenceHost(p).ok());

  gemm::GemmFp32Core(A.data(), B.data(), C_core.data(), M, N, K, 1.f, 0.f);
  EXPECT_TRUE(common::CheckEqual(C, C_core, 1e-6f));
  EXPECT_FLOAT_EQ(C[0], 19.f);
  EXPECT_FLOAT_EQ(C[3], 50.f);
}

TEST(GemmRefCpuTest, Fp32DeterministicFinite) {
  constexpr int M = 32, N = 64, K = 48;
  std::vector<float> A(static_cast<size_t>(M) * K);
  std::vector<float> B(static_cast<size_t>(K) * N);
  std::vector<float> C(static_cast<size_t>(M) * N);
  FillDeterministic(A, M, K, 0.3f);
  FillDeterministic(B, K, N, 0.2f);

  const auto p = MakeFp32Params(M, N, K, A, B, C);
  ASSERT_TRUE(gemm::GemmReferenceHost(p).ok());
  for (float v : C) {
    EXPECT_TRUE(std::isfinite(v));
  }
}

TEST(GemmRefCpuTest, RejectsFp16) {
  gemm::GemmParams p;
  p.M = p.N = p.K = 4;
  p.lda = p.ldb = p.ldc = 4;
  p.dtype_a = p.dtype_b = common::DType::kFp16;
  p.dtype_c = common::DType::kFp32;
  p.A = p.B = p.C = reinterpret_cast<void*>(0x1);
  const common::Status st = gemm::GemmReferenceHost(p);
  EXPECT_EQ(st.code, common::StatusCode::kUnsupported);
}

TEST(GemmRefCpuTest, CatalogPassCasesHost) {
  const auto cat = common::LoadTestCaseCatalog(
      std::string(CKO_SOURCE_DIR) + "/configs/test_cases/gemm.json");
  for (const auto& c : cat.cases) {
    if (c.expect != common::ExpectStatus::kPass) continue;
    if (c.gemm.dtype != "fp32") continue;
    if (SkipSlowCpuCatalogCase(c)) continue;

    const int M = c.gemm.M, N = c.gemm.N, K = c.gemm.K;
    std::vector<float> A(static_cast<size_t>(M) * K);
    std::vector<float> B(static_cast<size_t>(K) * N);
    std::vector<float> C(static_cast<size_t>(M) * N);
    FillDeterministic(A, M, K, 0.25f);
    FillDeterministic(B, K, N, 0.15f);

    const auto p = MakeFp32Params(M, N, K, A, B, C);
    EXPECT_TRUE(gemm::GemmReferenceHost(p).ok()) << "case id=" << c.id;
  }
}
