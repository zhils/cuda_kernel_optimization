#pragma once

#include <random>
#include <vector>

namespace rmsnorm {

// 生成随机测试配置: rows 和 cols 在 [128, 4096] 范围内
struct TestConfig {
  int rows;
  int cols;
};

inline TestConfig RandomTestConfig(uint32_t seed = 0) {
  static std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
  std::uniform_int_distribution<int> dist(128, 4096);
  return {dist(gen), dist(gen)};
}

// 生成 [-100, 100] 范围内的随机浮点数
inline std::vector<float> RandomMatrix(int rows, int cols, uint32_t seed = 0) {
  static std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  std::vector<float> mat(static_cast<size_t>(rows) * cols);
  for (auto& v : mat) v = dist(gen);
  return mat;
}

// 生成 [0.5, 1.5] 范围内的随机 weight (通常 weight 为正且接近 1)
inline std::vector<float> RandomWeight(int cols, uint32_t seed = 0) {
  static std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
  std::uniform_real_distribution<float> dist(0.5f, 1.5f);
  std::vector<float> w(cols);
  for (auto& v : w) v = dist(gen);
  return w;
}

}  // namespace rmsnorm