#include "common/test_case.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace common {
namespace {

ExpectStatus ParseExpectOne(const std::string& s) {
  if (s == "pass") return ExpectStatus::kPass;
  if (s == "skip") return ExpectStatus::kSkip;
  if (s == "invalid_argument") return ExpectStatus::kInvalidArgument;
  if (s == "unsupported") return ExpectStatus::kUnsupported;
  if (s == "fail") return ExpectStatus::kFail;
  throw std::runtime_error("unknown expect status: " + s);
}

GemmCaseParams ParseGemmParams(const nlohmann::json& j) {
  GemmCaseParams p;
  p.M = j.value("M", 0);
  p.N = j.value("N", 0);
  p.K = j.value("K", 0);
  p.dtype = j.value("dtype", "fp32");
  p.layout = j.value("layout", "row_major");
  p.impl = j.value("impl", "auto");
  return p;
}

RmsNormCaseParams ParseRmsNormParams(const nlohmann::json& j) {
  RmsNormCaseParams p;
  p.rows = j.value("rows", 0);
  p.cols = j.value("cols", 0);
  p.act_dtype = j.value("act_dtype", "fp32");
  p.weight_dtype = j.value("weight_dtype", "fp32");
  p.impl = j.value("impl", "auto");
  return p;
}

FusedCaseParams ParseFusedParams(const nlohmann::json& j) {
  FusedCaseParams p;
  p.B = j.value("B", 0);
  p.L = j.value("L", 0);
  p.D = j.value("D", 0);
  p.H = j.value("H", 0);
  p.k_size = j.value("k_size", 0);
  p.dtype = j.value("dtype", "fp32");
  p.impl = j.value("impl", "auto");
  return p;
}

TestCaseEntry ParseCaseEntry(const nlohmann::json& j, const std::string& family) {
  TestCaseEntry e;
  e.id = j.at("id").get<std::string>();
  e.family = family;
  e.expect = ParseExpectOne(j.at("expect").get<std::string>());
  if (j.contains("tags")) {
    for (const auto& t : j.at("tags")) {
      e.tags.push_back(t.get<std::string>());
    }
  }
  if (j.contains("params")) {
    const auto& p = j.at("params");
    if (family == "gemm") {
      e.gemm = ParseGemmParams(p);
    } else if (family == "rmsnorm") {
      e.rmsnorm = ParseRmsNormParams(p);
    } else if (family == "fused_conv1d_silu") {
      e.fused = ParseFusedParams(p);
    }
  }
  return e;
}

}  // namespace

ExpectStatus ParseExpectStatus(const std::string& s) { return ParseExpectOne(s); }

TestCaseCatalog LoadTestCaseCatalog(const std::string& json_path) {
  std::ifstream ifs(json_path);
  if (!ifs) {
    throw std::runtime_error("cannot open test catalog: " + json_path);
  }
  nlohmann::json root;
  ifs >> root;

  TestCaseCatalog cat;
  cat.family = root.at("family").get<std::string>();
  cat.schema_version = root.value("schema_version", "1");
  for (const auto& c : root.at("cases")) {
    cat.cases.push_back(ParseCaseEntry(c, cat.family));
  }
  return cat;
}

std::vector<TestCaseEntry> FilterByTag(const TestCaseCatalog& cat,
                                       const std::string& tag) {
  std::vector<TestCaseEntry> out;
  for (const auto& c : cat.cases) {
    for (const auto& t : c.tags) {
      if (t == tag) {
        out.push_back(c);
        break;
      }
    }
  }
  return out;
}

}  // namespace common
