#include <gtest/gtest.h>

#include "common/test_case.h"

#include <string>

#ifndef CKO_SOURCE_DIR
#define CKO_SOURCE_DIR "."
#endif

namespace {

std::string CatalogPath(const char* name) {
  return std::string(CKO_SOURCE_DIR) + "/configs/test_cases/" + name;
}

}  // namespace

TEST(CatalogLoaderTest, LoadGemmCatalog) {
  const auto cat = common::LoadTestCaseCatalog(CatalogPath("gemm.json"));
  EXPECT_EQ(cat.family, "gemm");
  EXPECT_GE(cat.cases.size(), 5u);
}

TEST(CatalogLoaderTest, FilterSmokeTag) {
  const auto cat = common::LoadTestCaseCatalog(CatalogPath("rmsnorm.json"));
  const auto smoke = common::FilterByTag(cat, "smoke");
  EXPECT_FALSE(smoke.empty());
  for (const auto& c : smoke) {
    bool has = false;
    for (const auto& t : c.tags) {
      if (t == "smoke") has = true;
    }
    EXPECT_TRUE(has);
  }
}

TEST(CatalogLoaderTest, InvalidExpectThrows) {
  EXPECT_THROW(common::ParseExpectStatus("not_a_status"), std::runtime_error);
}
