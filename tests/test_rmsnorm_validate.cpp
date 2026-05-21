#include <gtest/gtest.h>

#include "common/test_case.h"
#include "rmsnorm/rmsnorm_api.h"

#ifndef CKO_SOURCE_DIR
#define CKO_SOURCE_DIR "."
#endif

namespace {

rmsnorm::RmsNormParams ParamsFromCase(const common::TestCaseEntry& e) {
  rmsnorm::RmsNormParams p;
  p.rows = e.rmsnorm.rows;
  p.cols = e.rmsnorm.cols;
  common::ParseDType(e.rmsnorm.act_dtype.c_str(), &p.act_dtype);
  common::ParseDType(e.rmsnorm.weight_dtype.c_str(), &p.weight_dtype);
  return p;
}

}  // namespace

TEST(RmsNormValidateTest, CatalogExpectations) {
  const auto cat = common::LoadTestCaseCatalog(
      std::string(CKO_SOURCE_DIR) + "/configs/test_cases/rmsnorm.json");
  for (const auto& c : cat.cases) {
    const common::Status st = rmsnorm::ValidateRmsNormParams(ParamsFromCase(c), false);
    if (c.expect == common::ExpectStatus::kInvalidArgument) {
      EXPECT_EQ(st.code, common::StatusCode::kInvalidArgument) << c.id;
    } else {
      EXPECT_TRUE(st.ok()) << c.id << " msg=" << st.message;
    }
  }
}

TEST(RmsNormValidateTest, QuantRequiresScalesWhenRunning) {
  rmsnorm::RmsNormParams p;
  p.rows = 128;
  p.cols = 128;
  p.act_dtype = common::DType::kInt8;
  p.weight_dtype = common::DType::kFp32;
  p.input = reinterpret_cast<const void*>(0x1);
  p.weight = reinterpret_cast<const void*>(0x2);
  p.output = reinterpret_cast<void*>(0x3);
  const common::Status st = rmsnorm::ValidateRmsNormParams(p, true);
  EXPECT_EQ(st.code, common::StatusCode::kInvalidArgument);
}
