#pragma once

#include <functional>
#include <string>
#include <vector>

namespace test {

struct TestResult {
    std::string op_name;
    std::string version;
    int rows;
    int cols;
    bool passed;
    double max_abs_diff;
    double gpu_ms;
    double bandwidth_gb_s;
};

using OpTestFunc = std::function<void(const std::vector<float>&, std::vector<float>&, int, int)>;

void RegisterTest(const std::string& op_name, const std::string& version,
                  OpTestFunc cpu_func, OpTestFunc gpu_func,
                  int rows, int cols);

std::vector<TestResult> RunAllTests();

}  // namespace test
