#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <chrono>
#include <algorithm>
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
  constexpr int kWarmup = 3;
  constexpr int kRepeat = 10;
  constexpr int kMaxCpuVerifyDim = 1024;
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

    const bool do_cpu_verify = (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim && K <= kMaxCpuVerifyDim);
    double cpu_ms = 0.0;
    if (do_cpu_verify) {
      auto t0 = std::chrono::high_resolution_clock::now();
      GemmCPU(A.data(), B.data(), cpu.data(), M, N, K);
      auto t1 = std::chrono::high_resolution_clock::now();
      cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

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
    for (int w = 0; w < kWarmup; ++w) {
      CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K,
                               &beta, dC, N));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> cublas_times;
    cublas_times.reserve(kRepeat);
    for (int rep = 0; rep < kRepeat; ++rep) {
      CHECK_CUDA(cudaEventRecord(s));
      CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K,
                               &beta, dC, N));
      CHECK_CUDA(cudaEventRecord(e));
      CHECK_CUDA(cudaEventSynchronize(e));
      float ms = 0.f;
      CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
      cublas_times.push_back(ms);
    }
    std::sort(cublas_times.begin(), cublas_times.end());
    float cublas_ms = 0.f;
    if (cublas_times.size() > 2) {
      for (size_t t = 1; t + 1 < cublas_times.size(); ++t) {
        cublas_ms += cublas_times[t];
      }
      cublas_ms /= static_cast<float>(cublas_times.size() - 2);
    } else if (!cublas_times.empty()) {
      for (float t : cublas_times) cublas_ms += t;
      cublas_ms /= static_cast<float>(cublas_times.size());
    }
    CHECK_CUDA(cudaMemcpy(ref.data(), dC, ref.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    double max_abs_diff = 0.0;
    const char* check = "SKIP";
    if (do_cpu_verify) {
      ok = common::CheckEqual(cpu, ref, 1e-3f);
      max_abs_diff = common::MaxAbsDiff(cpu, ref);
      check = ok ? "PASS" : "FAIL";
    }
    ofs << cases[i].id << "," << cases[i].group << "," << M << "," << N << "," << K << "," << cpu_ms << "," << cublas_ms
        << "," << (cublas_ms > 0 ? cpu_ms / cublas_ms : 0) << "," << max_abs_diff << "," << check
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
