#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"
#include "rmsnorm/test_utils.h"
#include "rmsnorm_kernels.cuh"

// RMSNorm V0（朴素基线）
// 一个线程负责一整行：先算平方和，再做归一化并乘 weight。

static void RMSNormCPU(const float* x, float* y, const float* weight, int rows, int cols, float eps) {
  for (int r = 0; r < rows; ++r) {
    float sq_sum = 0.f;
    for (int c = 0; c < cols; ++c) {
      float val = x[r * cols + c];
      sq_sum += val * val;
    }
    float rms = 1.f / sqrtf(sq_sum / cols + eps);
    for (int c = 0; c < cols; ++c) {
      y[r * cols + c] = x[r * cols + c] * rms * weight[c];
    }
  }
}

int main() {
  constexpr float kEps = 1e-5f;
  constexpr int kTestCases = 5;
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/rmsnorm_naive_results.csv");
  ofs << "id,group,rows,cols,cpu_ms,gpu_ms,speedup,max_abs_diff,check\n";
  for (int i = 0; i < kTestCases; ++i) {
    auto cfg = rmsnorm::RandomTestConfig(2026 + i);
    int rows = cfg.rows, cols = cfg.cols, n = rows * cols;
    // ---------------- host 数据 + CPU 参考 ----------------
    std::vector<float> x = rmsnorm::RandomMatrix(rows, cols, 2026 + i);
    std::vector<float> weight = rmsnorm::RandomWeight(cols, 2026 + i + 100);
    std::vector<float> cpu(n), gpu(n);
    auto t0 = std::chrono::high_resolution_clock::now();
    RMSNormCPU(x.data(), cpu.data(), weight.data(), rows, cols, kEps);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---------------- 设备内存与执行 ----------------
    float *dx, *dy, *dweight;
    CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dweight, cols * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dweight, weight.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(s));
    RMSNormV0Kernel<<<(rows + 255) / 256, 256>>>(dx, dy, dweight, rows, cols, kEps);
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    float gpu_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, s, e));

    // ---------------- 回拷与记录 ----------------
    CHECK_CUDA(cudaMemcpy(gpu.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
    ofs << i << ",A," << rows << "," << cols << "," << cpu_ms << "," << gpu_ms << ","
        << (gpu_ms > 0 ? cpu_ms / gpu_ms : 0) << ","
        << common::MaxAbsDiff(cpu, gpu) << "," << (common::CheckEqual(cpu, gpu, 1e-4f) ? "PASS" : "FAIL") << "\n";

    // ---------------- 释放资源 ----------------
    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(dweight));
  }
  return 0;
}
