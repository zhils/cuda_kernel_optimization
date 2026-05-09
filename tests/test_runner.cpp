#include "test_utils.h"

#include <cuda_runtime.h>
#include <iostream>
#include <utility>

#include "common/cuda_utils.h"

namespace test {

namespace {
struct TestEntry {
    std::string op_name;
    std::string version;
    OpTestFunc cpu_func;
    OpTestFunc gpu_func;
    int rows;
    int cols;
};
static std::vector<TestEntry>& GetTestRegistry() {
    static std::vector<TestEntry> registry;
    return registry;
}
}  // namespace

void RegisterTest(const std::string& op_name, const std::string& version,
                 OpTestFunc cpu_func, OpTestFunc gpu_func,
                 int rows, int cols) {
    GetTestRegistry().push_back({op_name, version, std::move(cpu_func), std::move(gpu_func), rows, cols});
}

std::vector<TestResult> RunAllTests() {
    std::vector<TestResult> results;
    for (const auto& entry : GetTestRegistry()) {
        const int n = entry.rows * entry.cols;
        std::vector<float> input(n), cpu_out(n), gpu_out(n);

        for (float& v : input) {
            v = static_cast<float>(rand()) / RAND_MAX * 200.f - 100.f;
        }

        entry.cpu_func(input, cpu_out, entry.rows, entry.cols);

        float *d_in, *d_out;
        CHECK_CUDA(cudaMalloc(&d_in, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_in, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        cudaEvent_t start, end;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&end));
        CHECK_CUDA(cudaEventRecord(start));
        entry.gpu_func(d_in, d_out, entry.rows, entry.cols);
        CHECK_CUDA(cudaEventRecord(end));
        CHECK_CUDA(cudaEventSynchronize(end));

        float gpu_ms = 0.f;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, end));
        CHECK_CUDA(cudaMemcpy(gpu_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

        double max_diff = 0.0;
        bool passed = true;
        for (int i = 0; i < n; ++i) {
            double diff = std::abs(static_cast<double>(cpu_out[i]) - static_cast<double>(gpu_out[i]));
            if (diff > 1e-4) {
                passed = false;
            }
            if (diff > max_diff) max_diff = diff;
        }

        const double bytes = static_cast<double>(n) * sizeof(float) * 3.0;
        const double bw = bytes / (gpu_ms * 1e6);

        results.push_back({
            entry.op_name,
            entry.version,
            entry.rows,
            entry.cols,
            passed,
            max_diff,
            gpu_ms,
            bw
        });

        std::cout << entry.op_name << "_" << entry.version
                  << " | " << entry.rows << "x" << entry.cols
                  << " | " << gpu_ms << " ms"
                  << " | " << bw << " GB/s"
                  << " | " << (passed ? "PASS" : "FAIL") << "\n";

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(end));
        CHECK_CUDA(cudaFree(d_in));
        CHECK_CUDA(cudaFree(d_out));
    }
    return results;
}

}  // namespace test
