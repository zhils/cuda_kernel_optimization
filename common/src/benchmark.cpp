#include "common/benchmark.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

namespace common {

namespace {
void CreateDirectories(const std::string& path) {
  size_t pos = 0;
  std::string dir;
  while ((pos = path.find_first_of("/\\", pos)) != std::string::npos) {
    dir = path.substr(0, pos);
    if (!dir.empty()) {
#ifdef _WIN32
      _mkdir(dir.c_str());
#else
      mkdir(dir.c_str(), 0755);
#endif
    }
    pos++;
  }
}
}  // namespace

bool TryWriteProbe(const std::string& dir) {
  CreateDirectories(dir);
  const std::string probe = dir + "/.write_probe";
  std::ofstream out(probe);
  if (!out) {
    return false;
  }
  out << "ok";
  out.close();
  std::remove(probe.c_str());
  return true;
}

std::string EnsureResultsDir() {
  if (const char* env = std::getenv("CUDA_RESULTS_DIR")) {
    if (*env && TryWriteProbe(env)) {
      return env;
    }
  }
  for (const char* candidate : {"data/results", "build/data/results", "results"}) {
    if (TryWriteProbe(candidate)) {
      return candidate;
    }
  }
  return "results";
}

std::vector<TestCase> BuildDefaultTestCases(int min_dim, int max_dim, int total_cases,
                                            int max_elements, uint32_t seed) {
  (void)min_dim;
  (void)max_dim;
  (void)total_cases;
  (void)max_elements;
  (void)seed;

  // 组 A：小规模
  // 组 B：中等规模
  // 组 C：较大规模
  // 组 D：大规模
  // 组 E：超大规模
  return {
      {0, "A", 128, 128},
      {1, "A", 256, 256},
      {2, "B", 512, 512},
      {3, "B", 1024, 1024},
      {4, "C", 2048, 2048},
      {5, "D", 4096, 4096},
      {6, "E", 8192, 8192},
  };
}

bool WriteTestCasesCsv(const std::string& file_path,
                       const std::vector<TestCase>& test_cases) {
  CreateDirectories(file_path);

  // 打开文件写入表头
  std::ofstream ofs(file_path, std::ios::out | std::ios::trunc);
  if (!ofs.is_open()) {
    return false;
  }

  ofs << "id,group,rows,cols\n";

  // 逐行写入测试用例
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

  // 跳过表头
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

    // 解析 CSV 行数据
    std::stringstream ss(line);
    std::string cell;
    std::vector<std::string> items;
    while (std::getline(ss, cell, ',')) {
      items.push_back(cell);
    }
    if (items.size() < 4) {
      continue;
    }

    // 构造测试用例
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
  // 尝试从文件加载
  auto cases = LoadTestCasesCsv(file_path);
  auto defaults = BuildDefaultTestCases();

  // 加载失败则使用默认测试用例
  if (!cases.empty() && cases.size() == defaults.size()) {
    return cases;
  }
  return defaults;
}

void InitMatrix(std::vector<float>& matrix, int rows, int cols) {
  // 按行列生成确定性伪随机值
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      matrix[static_cast<size_t>(r) * cols + c] =
          static_cast<float>((r * 131 + c * 17) % 1000) * 0.001f;
    }
  }
}

bool CheckEqual(const std::vector<float>& a, const std::vector<float>& b, float eps) {
  // 长度校验
  if (a.size() != b.size()) {
    return false;
  }

  // 逐元素误差比较
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::fabs(a[i] - b[i]) > eps) {
      return false;
    }
  }
  return true;
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  // 长度校验
  if (a.size() != b.size()) {
    return std::numeric_limits<double>::infinity();
  }

  // 逐元素计算最大绝对差
  double max_diff = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, static_cast<double>(std::fabs(a[i] - b[i])));
  }
  return max_diff;
}

double MaxAbsDiff(const std::vector<int>& a, const std::vector<int>& b) {
  // 长度校验
  if (a.size() != b.size()) {
    return std::numeric_limits<double>::infinity();
  }

  // 逐元素计算最大绝对差
  double max_diff = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, static_cast<double>(std::abs(a[i] - b[i])));
  }
  return max_diff;
}

}  // namespace common
