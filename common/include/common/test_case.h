#pragma once

#include <string>
#include <vector>

namespace common {

// Expected outcome for catalog-driven tests.
enum class ExpectStatus : int {
  kPass = 0,
  kSkip = 1,
  kInvalidArgument = 2,
  kUnsupported = 3,
  kFail = 4,
};

struct GemmCaseParams {
  int M = 0;
  int N = 0;
  int K = 0;
  std::string dtype = "fp32";
  std::string layout = "row_major";
  std::string impl = "auto";
};

struct RmsNormCaseParams {
  int rows = 0;
  int cols = 0;
  std::string act_dtype = "fp32";
  std::string weight_dtype = "fp32";
  std::string impl = "auto";
};

struct FusedCaseParams {
  int B = 0;
  int L = 0;
  int D = 0;
  int H = 0;
  int k_size = 0;
  std::string dtype = "fp32";
  std::string impl = "auto";
};

struct TestCaseEntry {
  std::string id;
  std::string family;
  ExpectStatus expect = ExpectStatus::kPass;
  std::vector<std::string> tags;
  GemmCaseParams gemm;
  RmsNormCaseParams rmsnorm;
  FusedCaseParams fused;
};

struct TestCaseCatalog {
  std::string family;
  std::string schema_version;
  std::vector<TestCaseEntry> cases;
};

ExpectStatus ParseExpectStatus(const std::string& s);

// Load configs/test_cases/<family>.json relative to project root or explicit path.
TestCaseCatalog LoadTestCaseCatalog(const std::string& json_path);

std::vector<TestCaseEntry> FilterByTag(const TestCaseCatalog& cat,
                                         const std::string& tag);

}  // namespace common
