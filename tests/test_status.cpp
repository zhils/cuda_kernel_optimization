#include <gtest/gtest.h>

#include "common/status.h"

TEST(StatusTest, OkIsSuccess) {
  const common::Status st = common::Status::Ok();
  EXPECT_TRUE(st.ok());
  EXPECT_EQ(common::StatusCodeName(st.code), std::string("ok"));
}

TEST(StatusTest, StatusToCMatches) {
  EXPECT_EQ(common::StatusToC(common::StatusCode::kInvalidArgument), CKO_INVALID_ARGUMENT);
  EXPECT_EQ(common::StatusToC(common::StatusCode::kUnimplemented), CKO_UNIMPLEMENTED);
}
