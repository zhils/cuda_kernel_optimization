#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <chrono>
#include <filesystem>
#include <fstream>
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
  auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/gemm_nvidia_ref_results.csv");
  ofs << "id,group,M,N,K,cpu_ms,cublas_ms,speedup,max_abs_diff,check\n";

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  for (size_t i = 0; i < cases.size(); ++i) {
    int M = cases[i].rows, N = cases[i].cols, K = (M + N) / 2;
    std::vector<float> A(M * K), B(K * N), cpu(M * N), ref(M * N);
    common::InitMatrix(A, M, K);
    common::InitMatrix(B, K, N);

    auto t0 = std::chrono::high_resolution_clock::now();
    GemmCPU(A.data(), B.data(), cpu.data(), M, N, K);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, A.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, B.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, ref.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

    // cuBLAS uses column-major; we use transposed ops to match row-major GEMM.
    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(s));
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K,
                             &beta, dC, N));
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));

    float cublas_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&cublas_ms, s, e));
    CHECK_CUDA(cudaMemcpy(ref.data(), dC, ref.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = common::CheckEqual(cpu, ref, 1e-3f);
    ofs << cases[i].id << "," << cases[i].group << "," << M << "," << N << "," << K << "," << cpu_ms << "," << cublas_ms
        << "," << (cublas_ms > 0 ? cpu_ms / cublas_ms : 0) << "," << common::MaxAbsDiff(cpu, ref) << "," << (ok ? "PASS" : "FAIL")
        << "\n";

    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
  }
  CHECK_CUBLAS(cublasDestroy(handle));
  return 0;
}
