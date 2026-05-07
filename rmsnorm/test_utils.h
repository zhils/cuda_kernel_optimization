#pragma once

#include <random>
#include <vector>

namespace rmsnorm {

struct TestConfig {
  int rows;
  int cols;
};

inline TestConfig RandomTestConfig(uint32_t seed = 0) {
  static std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
  std::uniform_int_distribution<int> dist(128, 4096);
  int rows = dist(gen);
  int cols = (dist(gen) + 3) / 4 * 4;
  return {rows, cols};
}

inline std::vector<float> RandomMatrix(int rows, int cols, uint32_t seed = 0) {
  static std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  std::vector<float> mat(static_cast<size_t>(rows) * cols);
  for (auto& v : mat) v = dist(gen);
  return mat;
}

inline std::vector<float> RandomWeight(int cols, uint32_t seed = 0) {
  static std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
  std::uniform_real_distribution<float> dist(0.5f, 1.5f);
  std::vector<float> w(cols);
  for (auto& v : w) v = dist(gen);
  return w;
}

}  // namespace rmsnorm
