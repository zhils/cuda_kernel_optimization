#include "common/benchmark.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>

namespace common {

std::vector<TestCase> BuildDefaultTestCases(int min_dim, int max_dim, int total_cases,
                                            int max_elements, uint32_t seed) {
  (void)min_dim;
  (void)max_dim;
  (void)total_cases;
  (void)max_elements;
  (void)seed;
  return {
      {0, "A", 32, 32},
      {1, "A", 128, 256},
      {2, "B", 1111, 222},
      {3, "C", 4096, 4096},
      {4, "D", 32768, 64},
  };
}

bool WriteTestCasesCsv(const std::string& file_path,
                       const std::vector<TestCase>& test_cases) {
  std::filesystem::path path(file_path);
  std::filesystem::create_directories(path.parent_path());

  std::ofstream ofs(file_path, std::ios::out | std::ios::trunc);
  if (!ofs.is_open()) {
    return false;
  }

  ofs << "id,group,rows,cols\n";
  for (size_t i = 0; i < test_cases.size(); ++i) {
    ofs << test_cases[i].id << "," << test_cases[i].group << ","
        << test_cases[i].rows << "," << test_cases[i].cols << "\n";
  }
  return true;
}

std::vector<TestCase> LoadTestCasesCsv(const std::string& file_path) {
  std::vector<TestCase> cases;
  std::ifstream ifs(file_path);
  if (!ifs.is_open()) {
    return cases;
  }

  std::string line;
  bool header_skipped = false;
  int id = 0;
  while (std::getline(ifs, line)) {
    if (line.empty()) {
      continue;
    }
    if (!header_skipped) {
      header_skipped = true;
      continue;
    }

    std::stringstream ss(line);
    std::string cell;
    std::vector<std::string> items;
    while (std::getline(ss, cell, ',')) {
      items.push_back(cell);
    }
    if (items.size() < 4) {
      continue;
    }

    try {
      TestCase tc{};
      tc.id = std::stoi(items[0]);
      tc.group = items[1];
      tc.rows = std::stoi(items[2]);
      tc.cols = std::stoi(items[3]);
      cases.push_back(tc);
    } catch (...) {
      TestCase tc{};
      tc.id = id++;
      tc.group = "A";
      tc.rows = std::stoi(items[0]);
      tc.cols = std::stoi(items[1]);
      cases.push_back(tc);
    }
  }
  return cases;
}

std::vector<TestCase> LoadOrCreateTestCasesCsv(const std::string& file_path) {
  auto cases = LoadTestCasesCsv(file_path);
  if (!cases.empty()) {
    return cases;
  }

  cases = BuildDefaultTestCases();
  WriteTestCasesCsv(file_path, cases);
  return cases;
}

void InitMatrix(std::vector<float>& matrix, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      matrix[static_cast<size_t>(r) * cols + c] =
          static_cast<float>((r * 131 + c * 17) % 1000) * 0.001f;
    }
  }
}

bool CheckEqual(const std::vector<float>& a, const std::vector<float>& b, float eps) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::fabs(a[i] - b[i]) > eps) {
      return false;
    }
  }
  return true;
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double max_diff = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, static_cast<double>(std::fabs(a[i] - b[i])));
  }
  return max_diff;
}

double MaxAbsDiff(const std::vector<int>& a, const std::vector<int>& b) {
  if (a.size() != b.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double max_diff = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, static_cast<double>(std::abs(a[i] - b[i])));
  }
  return max_diff;
}

}  // namespace common