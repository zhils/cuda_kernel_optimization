#include <cuda_runtime.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../common/include/common/benchmark.h"
#include "../common/include/common/cuda_utils.h"
// CPU 参考实现定义见下方

static void RMSNormCPU(const float* x, const float* weight, float* y, int rows, int cols, float eps) {
  for (int i = 0; i < rows; ++i) {
    double ss = 0;
    for (int j = 0; j < cols; ++j) ss += (double)x[i * cols + j] * x[i * cols + j];
    double rms = sqrt(ss / cols + (double)eps);
    for (int j = 0; j < cols; ++j) y[i * cols + j] = x[i * cols + j] / (float)rms * weight[j];
  }
}

#include "rmsnorm/test_utils.h"
#include "rmsnorm_kernels.cuh"

// RMSNorm V0：朴素基线，一个线程负责一整行，先算平方和，再做归一化并乘 weight

int main() {
  constexpr float kEps = 1e-5f;
  constexpr int kRepeat = 10;
  constexpr int kTestCases = 5;

  const std::string results_dir = common::EnsureResultsDir();
  std::ofstream ofs(results_dir + "/rmsnorm_v0_results.csv");
  ofs << "id,rows,cols,gpu_ms,bandwidth_gb_s,max_abs_diff,check\n";

  const std::vector<std::pair<int, int>> test_sizes = {
      {128, 128}, {256, 256}, {512, 512}, {1024, 1024}, {4096, 4096}};

  for (int i = 0; i < kTestCases; ++i) {
    // 确定维度并生成测试数据
    const int rows = test_sizes[i].first;
    const int cols = test_sizes[i].second;
    const int n = rows * cols;

    std::vector<float> x = rmsnorm::RandomMatrix(rows, cols, 2026 + i);
    std::vector<float> weight = rmsnorm::RandomWeight(cols, 2026 + i + 100);
    std::vector<float> cpu(n), gpu(n);
    RMSNormCPU(x.data(), weight.data(), cpu.data(), rows, cols, kEps);

    // 分配 GPU 内存
    float *dx, *dy, *dweight;
    CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dweight, cols * sizeof(float)));

    // 拷贝数据到 GPU
    CHECK_CUDA(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dweight, weight.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    // 预热
    const dim3 grid((rows + 255) / 256);
    const dim3 block(256);

    RMSNormV0Kernel<<<grid, block>>>(dx, dy, dweight, rows, cols, kEps);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 计时循环
    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(s));
    for (int rep = 0; rep < kRepeat; ++rep) {
      RMSNormV0Kernel<<<grid, block>>>(dx, dy, dweight, rows, cols, kEps);
    }
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    float gpu_ms_total = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_total, s, e));
    const float gpu_ms = gpu_ms_total / static_cast<float>(kRepeat);

    // 拷贝结果回 CPU
    CHECK_CUDA(cudaMemcpy(gpu.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));

    // 校验与结果输出
    const bool ok = common::CheckEqual(cpu, gpu, 1e-4f);

    const double bytes = static_cast<double>(n) * sizeof(float) * 2.0;
    const double bw = bytes / (static_cast<double>(gpu_ms) * 1e6);

    std::cout << rows << "x" << cols << " | " << std::fixed << std::setprecision(4) << gpu_ms
              << " ms | " << std::setprecision(1) << bw << " GB/s | " << (ok ? "PASS" : "FAIL")
              << "\n";

    ofs << i << "," << rows << "," << cols << "," << gpu_ms << "," << bw << ","
        << common::MaxAbsDiff(cpu, gpu) << "," << (ok ? "PASS" : "FAIL") << "\n";

    // 释放资源
    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(dweight));
  }
  return 0;
}
