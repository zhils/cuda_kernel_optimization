// RMSNorm V2: 在 V1 基础上, 使用 Warp 归约计算 sq_sum

#include <cuda_runtime.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../common/include/common/benchmark.h"
#include "../common/include/common/cuda_utils.h"
#include "rmsnorm/test_utils.h"
#include "rmsnorm_kernels.cuh"

#define EPS 1e-5f

static void RMSNormCPU(const float* x, float* y, const float* weight, int rows, int cols,
                       float eps) {
  for (int r = 0; r < rows; ++r) {
    float sq_sum = 0.f;
    for (int c = 0; c < cols; ++c) {
      float val = x[r * cols + c];
      sq_sum += val * val;
    }
    float rms = 1.f / sqrtf(sq_sum / cols + eps);
    for (int c = 0; c < cols; ++c)
      y[r * cols + c] = x[r * cols + c] * rms * weight[c];
  }
}

int main() {
  constexpr int kRepeat = 10;
  constexpr int kTestCases = 5;
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/rmsnorm_v2_results.csv");
  ofs << "id,rows,cols,gpu_ms,bandwidth_gb_s,max_abs_diff,check\n";

  cudaDeviceProp dev_prop;
  CHECK_CUDA(cudaGetDeviceProperties(&dev_prop, 0));
  if (dev_prop.major >= 12) {
    CHECK_CUDA(
        cudaFuncSetAttribute(RMSNormV2Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
  }

  for (int i = 0; i < kTestCases; ++i) {
    auto cfg = rmsnorm::RandomTestConfig(2026 + i);
    int rows = cfg.rows, cols = cfg.cols, n = rows * cols;
    std::vector<float> x = rmsnorm::RandomMatrix(rows, cols, 2026 + i);
    std::vector<float> w = rmsnorm::RandomWeight(cols, 2026 + i + 100);
    std::vector<float> cpu(n), gpu(n);
    RMSNormCPU(x.data(), cpu.data(), w.data(), rows, cols, EPS);

    float *dx, *dy, *dw;
    CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dw, cols * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw, w.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    const size_t smem_size = RMSNORM_WARPS_PER_BLOCK * cols * sizeof(float);
    CHECK_CUDA(cudaFuncSetAttribute(RMSNormV2Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 block(RMSNORM_BLOCK_SIZE);
    dim3 grid((rows + RMSNORM_WARPS_PER_BLOCK - 1) / RMSNORM_WARPS_PER_BLOCK);
    RMSNormV2Kernel<<<grid, block, smem_size>>>(dx, dy, dw, rows, cols, EPS);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(s));
    for (int rep = 0; rep < kRepeat; ++rep) {
      RMSNormV2Kernel<<<grid, block, smem_size>>>(dx, dy, dw, rows, cols, EPS);
    }
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    float gpu_ms_total = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_total, s, e));
    const float gpu_ms = gpu_ms_total / static_cast<float>(kRepeat);

    CHECK_CUDA(cudaMemcpy(gpu.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = common::CheckEqual(cpu, gpu, 1e-4f);

    const double bytes = static_cast<double>(n) * sizeof(float) * 3.0;
    const double bw = bytes / (static_cast<double>(gpu_ms) * 1e6);

    std::cout << rows << "x" << cols
              << " | " << std::fixed << std::setprecision(4) << gpu_ms << " ms"
              << " | " << std::setprecision(1) << bw << " GB/s"
              << " | " << (ok ? "PASS" : "FAIL") << "\n";

    ofs << i << "," << rows << "," << cols << "," << gpu_ms << "," << bw << ","
        << common::MaxAbsDiff(cpu, gpu) << "," << (ok ? "PASS" : "FAIL") << "\n";

    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(dw));
  }
  return 0;
}
