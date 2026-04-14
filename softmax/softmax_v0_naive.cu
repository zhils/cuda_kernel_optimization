#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

__global__ void SoftmaxNaiveKernel(const float* x, float* y, int rows, int cols) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float maxv = x[r * cols];
  for (int c = 1; c < cols; ++c) maxv = fmaxf(maxv, x[r * cols + c]);
  float sum = 0.f;
  for (int c = 0; c < cols; ++c) {
    float v = expf(x[r * cols + c] - maxv);
    y[r * cols + c] = v;
    sum += v;
  }
  for (int c = 0; c < cols; ++c) y[r * cols + c] /= sum;
}

static void SoftmaxCPU(const float* x, float* y, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    float maxv = x[r * cols];
    for (int c = 1; c < cols; ++c) maxv = std::max(maxv, x[r * cols + c]);
    float sum = 0.f;
    for (int c = 0; c < cols; ++c) {
      float v = std::exp(x[r * cols + c] - maxv);
      y[r * cols + c] = v;
      sum += v;
    }
    for (int c = 0; c < cols; ++c) y[r * cols + c] /= sum;
  }
}

int main() {
  auto cases = common::LoadOrCreateTestCasesCsv("data/softmax/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/softmax_naive_results.csv");
  ofs << "id,group,rows,cols,cpu_ms,gpu_ms,speedup,max_abs_diff,check\n";
  for (size_t i = 0; i < cases.size(); ++i) {
    int rows = cases[i].rows, cols = cases[i].cols, n = rows * cols;
    std::vector<float> x(n), cpu(n), gpu(n);
    common::InitMatrix(x, rows, cols);
    auto t0 = std::chrono::high_resolution_clock::now();
    SoftmaxCPU(x.data(), cpu.data(), rows, cols);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    float *dx, *dy;
    CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(s));
    SoftmaxNaiveKernel<<<(rows + 255) / 256, 256>>>(dx, dy, rows, cols);
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    CHECK_CUDA(cudaGetLastError());
    float gpu_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, s, e));
    CHECK_CUDA(cudaMemcpy(gpu.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = common::CheckEqual(cpu, gpu, 1e-4f);
    ofs << i << "," << cases[i].group << "," << rows << "," << cols << "," << cpu_ms << "," << gpu_ms << ","
        << (gpu_ms > 0 ? cpu_ms / gpu_ms : 0) << "," << common::MaxAbsDiff(cpu, gpu) << "," << (ok ? "PASS" : "FAIL") << "\n";
    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
  }
  return 0;
}
