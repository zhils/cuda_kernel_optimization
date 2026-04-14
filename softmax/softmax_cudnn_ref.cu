#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

#define CHECK_CUDNN(call)                                                        \
  do {                                                                            \
    cudnnStatus_t s__ = (call);                                                  \
    if (s__ != CUDNN_STATUS_SUCCESS) {                                           \
      std::cerr << "cuDNN error: " << s__ << std::endl;                          \
      std::exit(EXIT_FAILURE);                                                    \
    }                                                                            \
  } while (0)

static void SoftmaxCPU(const float* x, float* y, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    float max_val = -INFINITY;
    for (int c = 0; c < cols; ++c) {
      max_val = fmaxf(max_val, x[r * cols + c]);
    }
    float sum = 0.f;
    for (int c = 0; c < cols; ++c) {
      sum += expf(x[r * cols + c] - max_val);
    }
    for (int c = 0; c < cols; ++c) {
      y[r * cols + c] = expf(x[r * cols + c] - max_val) / sum;
    }
  }
}

int main() {
  auto cs = common::LoadOrCreateTestCasesCsv("data/softmax/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/softmax_nvidia_ref_results.csv");
  ofs << "id,group,rows,cols,cpu_ms,gpu_naive_ms,gpu_cudnn_ms,speedup_naive,speedup_cudnn,max_abs_diff,check\n";

  cudnnHandle_t cudnn;
  CHECK_CUDNN(cudnnCreate(&cudnn));

  for (size_t i = 0; i < cs.size(); ++i) {
    int rows = cs[i].rows, cols = cs[i].cols, n = rows * cols;
    std::vector<float> x(n), cpu(n), ref_gpu(n);
    common::InitMatrix(x, rows, cols);

    auto t0 = std::chrono::high_resolution_clock::now();
    SoftmaxCPU(x.data(), cpu.data(), rows, cols);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    float *dx, *dy;
    CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cudnnTensorDescriptor_t desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, rows, cols, 1, 1));

    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));

    float alpha = 1.f, beta = 0.f;
    CHECK_CUDA(cudaEventRecord(s));
    CHECK_CUDNN(cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                      &alpha, desc, dx, &beta, desc, dy));
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    float cudnn_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&cudnn_ms, s, e));
    CHECK_CUDA(cudaMemcpy(ref_gpu.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));

    ofs << i << "," << cs[i].group << "," << rows << "," << cols << "," << cpu_ms << "," << 0 << "," << cudnn_ms << ","
        << 0 << ","
        << (cudnn_ms > 0 ? cpu_ms / cudnn_ms : 0) << ","
        << common::MaxAbsDiff(cpu, ref_gpu) << "," << (common::CheckEqual(cpu, ref_gpu, 1e-4f) ? "PASS" : "FAIL") << "\n";

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(desc));
    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
  }
  CHECK_CUDNN(cudnnDestroy(cudnn));
  return 0;
}