#include <gtest/gtest.h>

#include "common/test_case.h"
#include "gemm/gemm_api.h"

#ifndef CKO_SOURCE_DIR
#define CKO_SOURCE_DIR "."
#endif

namespace {

gemm::GemmParams ParamsFromCase(const common::TestCaseEntry& e) {
  gemm::GemmParams p;
  p.M = e.gemm.M;
  p.N = e.gemm.N;
  p.K = e.gemm.K;
  p.lda = p.K;
  p.ldb = p.N;
  p.ldc = p.N;
  common::ParseDType(e.gemm.dtype.c_str(), &p.dtype_a);
  p.dtype_b = p.dtype_a;
  p.dtype_c = common::DType::kFp32;
  p.impl = gemm::ImplId::kAuto;
  common::Status note;
  p.impl = gemm::ResolveImpl(p, &note);
  return p;
}

common::ExpectStatus ValidateCatalogCase(const common::TestCaseEntry& e) {
  const common::Status st = gemm::ValidateGemmParams(ParamsFromCase(e), false);
  switch (e.expect) {
    case common::ExpectStatus::kPass:
      return st.ok() ? common::ExpectStatus::kPass : common::ExpectStatus::kFail;
    case common::ExpectStatus::kSkip:
      return st.ok() ? common::ExpectStatus::kSkip : common::ExpectStatus::kFail;
    case common::ExpectStatus::kInvalidArgument:
      return st.code == common::StatusCode::kInvalidArgument
                 ? common::ExpectStatus::kInvalidArgument
                 : common::ExpectStatus::kFail;
    case common::ExpectStatus::kUnsupported:
      return st.code == common::StatusCode::kUnsupported
                 ? common::ExpectStatus::kUnsupported
                 : common::ExpectStatus::kFail;
    default:
      return common::ExpectStatus::kFail;
  }
}

}  // namespace

TEST(GemmValidateTest, RejectsZeroDimension) {
  gemm::GemmParams p;
  p.M = 0;
  p.N = 128;
  p.K = 128;
  p.lda = p.K;
  p.ldb = p.N;
  p.ldc = p.N;
  const common::Status st = gemm::ValidateGemmParams(p, false);
  EXPECT_EQ(st.code, common::StatusCode::kInvalidArgument);
}

TEST(GemmValidateTest, UnalignedFallbackToCublas) {
  gemm::GemmParams p;
  p.M = 1000;
  p.N = 1000;
  p.K = 1000;
  p.impl = gemm::ImplId::kV3;
  p.alignment_policy = gemm::AlignmentPolicy::kFallback;
  common::Status note;
  const gemm::ImplId resolved = gemm::ResolveImpl(p, &note);
  EXPECT_EQ(resolved, gemm::ImplId::kCublas);
}

TEST(GemmValidateTest, CatalogExpectations) {
  const auto cat = common::LoadTestCaseCatalog(
      std::string(CKO_SOURCE_DIR) + "/configs/test_cases/gemm.json");
  for (const auto& c : cat.cases) {
    EXPECT_EQ(ValidateCatalogCase(c), c.expect) << "case id=" << c.id;
  }
}

TEST(GemmValidateTest, UnalignedFp16FallbackToCublasFp16) {
  gemm::GemmParams p;
  p.M = 1000;
  p.N = 1000;
  p.K = 1000;
  p.dtype_a = p.dtype_b = common::DType::kFp16;
  p.dtype_c = common::DType::kFp32;
  p.impl = gemm::ImplId::kFp16;
  p.alignment_policy = gemm::AlignmentPolicy::kFallback;
  common::Status note;
  const gemm::ImplId resolved = gemm::ResolveImpl(p, &note);
  EXPECT_EQ(resolved, gemm::ImplId::kCublasFp16);
}
