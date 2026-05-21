#include <gtest/gtest.h>

#include "common/test_case.h"
#include "fused/fused_api.h"

#ifndef CKO_SOURCE_DIR
#define CKO_SOURCE_DIR "."
#endif

namespace {

fused::FusedParams ParamsFromCase(const common::TestCaseEntry& e) {
  fused::FusedParams p;
  p.B = e.fused.B;
  p.L = e.fused.L;
  p.D = e.fused.D;
  p.H = e.fused.H;
  p.k_size = e.fused.k_size;
  return p;
}

}  // namespace

TEST(FusedValidateTest, CatalogExpectations) {
  const auto cat = common::LoadTestCaseCatalog(
      std::string(CKO_SOURCE_DIR) + "/configs/test_cases/fused_conv1d_silu.json");
  for (const auto& c : cat.cases) {
    const common::Status st = fused::ValidateFusedParams(ParamsFromCase(c), false);
    if (c.expect == common::ExpectStatus::kInvalidArgument) {
      EXPECT_EQ(st.code, common::StatusCode::kInvalidArgument) << c.id;
    } else {
      EXPECT_TRUE(st.ok()) << c.id << " msg=" << st.message;
    }
  }
}

TEST(FusedValidateTest, KSizeCannotExceedL) {
  fused::FusedParams p;
  p.B = 1;
  p.L = 4;
  p.D = 16;
  p.H = 8;
  p.k_size = 8;
  const common::Status st = fused::ValidateFusedParams(p, false);
  EXPECT_EQ(st.code, common::StatusCode::kInvalidArgument);
}
