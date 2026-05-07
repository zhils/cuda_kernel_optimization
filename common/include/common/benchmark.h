#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace common {

struct TestCase {
  int id;
  std::string group;
  int rows;
  int cols;
};

std::vector<TestCase> BuildDefaultTestCases(int min_dim = 32, int max_dim = 4096,
                                            int total_cases = 5,
                                            int max_elements = (1 << 20),
                                            uint32_t seed = 20260323);

bool WriteTestCasesCsv(const std::string& file_path,
                       const std::vector<TestCase>& test_cases);

std::vector<TestCase> LoadTestCasesCsv(const std::string& file_path);

std::vector<TestCase> LoadOrCreateTestCasesCsv(const std::string& file_path);

void InitMatrix(std::vector<float>& matrix, int rows, int cols);

bool CheckEqual(const std::vector<float>& a, const std::vector<float>& b,
                float eps = 1e-6f);

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b);

double MaxAbsDiff(const std::vector<int>& a, const std::vector<int>& b);

}  // namespace common