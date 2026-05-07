#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

#define CHECK_CUBLAS(call)                                                       \
  do {                                                                           \
    cublasStatus_t s__ = (call);                                                 \
    if (s__ != CUBLAS_STATUS_SUCCESS) {                                          \
      std::cerr << "cuBLAS error: " << static_cast<int>(s__) << " at "           \
                << __FILE__ << ":" << __LINE__ << std::endl;                    \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

static void GemmCPU(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < N; ++c) {
      float s = 0;
      for (int k = 0; k < K; ++k) s += A[r * K + k] * B[k * N + c];
      C[r * N + c] = s;
    }
}

int main() {
  constexpr int kWarmup = 3;
  constexpr int kRepeat = 10;
  constexpr int kMaxCpuVerifyDim = 1024;
  auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/gemm_cublas_tensor_results.csv");
  ofs << "id,group,M,N,K,gpu_ms,gflops,max_abs_diff,check\n";

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  std::cout << "cuBLAS SGEMM (FP32 input, FP32 accumulate)\n";
  std::cout << "===========================================\n\n";

  for (size_t i = 0; i < cases.size(); ++i) {
    int M = cases[i].rows, N = cases[i].cols, K = M;
    std::vector<float> A(M * K), B(K * N), cpu(M * N);
    common::InitMatrix(A, M, K);
    common::InitMatrix(B, K, N);

    const bool do_cpu_verify = (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim);
    if (do_cpu_verify) {
      GemmCPU(A.data(), B.data(), cpu.data(), M, N, K);
    }

    float *dA, *dB;
    float *dC;
    CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, M * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;

    for (int w = 0; w < kWarmup; ++w) {
      CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K,
                               &alpha,
                               dB, N,
                               dA, K,
                               &beta,
                               dC, N));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(s));
    for (int rep = 0; rep < kRepeat; ++rep) {
      CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K,
                               &alpha,
                               dB, N,
                               dA, K,
                               &beta,
                               dC, N));
    }
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
    ms /= kRepeat;

    std::vector<float> ref(M * N);
    CHECK_CUDA(cudaMemcpy(ref.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    double max_abs_diff = 0.0;
    const char* check = "SKIP";
    if (do_cpu_verify) {
      ok = common::CheckEqual(cpu, ref, 1e-2f);
      max_abs_diff = common::MaxAbsDiff(cpu, ref);
      check = ok ? "PASS" : "FAIL";
    }

    double gflops = (2.0 * M * N * K) / (ms * 1e6);

    std::cout << M << "x" << N << "x" << K << " | " << std::fixed << std::setprecision(4) << ms << " ms"
              << " | " << std::setprecision(1) << gflops << " GFLOP/s"
              << " | " << check << "\n";

    ofs << i << ",cublas_tensor," << M << "," << N << "," << K << ","
        << ms << "," << gflops << ","
        << max_abs_diff << "," << check << "\n";

    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
  }

  CHECK_CUBLAS(cublasDestroy(handle));
  return 0;
}
