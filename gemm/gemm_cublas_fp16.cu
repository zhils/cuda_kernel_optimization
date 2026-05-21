#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
  // 参数与输出文件准备
  constexpr int kWarmup = 3;
  constexpr int kRepeat = 10;
  constexpr int kMaxCpuVerifyDim = 1024;
  auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/gemm_cublas_fp16_results.csv");
  ofs << "id,group,M,N,K,ms,gflops,max_abs_diff,check\n";

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  std::cout << "cuBLAS HGEMM (FP16 input, FP32 accumulate, Tensor Core)\n";
  std::cout << "========================================================\n\n";

  for (size_t i = 0; i < cases.size(); ++i) {
    int M = cases[i].rows, N = cases[i].cols, K = M;

    // 生成测试数据
    std::vector<float> A_fp32(M * K), B_fp32(K * N), cpu(M * N);
    common::InitMatrix(A_fp32, M, K);
    common::InitMatrix(B_fp32, K, N);

    // CPU参考计算
    const bool do_cpu_verify = (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim);
    if (do_cpu_verify) {
      GemmCPU(A_fp32.data(), B_fp32.data(), cpu.data(), M, N, K);
    }

    // 转换为FP16
    std::vector<__half> A_half(M * K), B_half(K * N);
    for (size_t j = 0; j < A_half.size(); ++j) A_half[j] = __float2half(A_fp32[j]);
    for (size_t j = 0; j < B_half.size(); ++j) B_half[j] = __float2half(B_fp32[j]);

    // 分配GPU内存并拷贝数据
    __half *dA, *dB;
    float *dC;
    CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC, M * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, A_half.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B_half.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;

    // 预热
    for (int w = 0; w < kWarmup; ++w) {
      CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K,
                                &alpha,
                                dB, CUDA_R_16F, N,
                                dA, CUDA_R_16F, K,
                                &beta,
                                dC, CUDA_R_32F, N,
                                CUBLAS_COMPUTE_32F_FAST_16F,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // 计时循环
    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(s));
    for (int rep = 0; rep < kRepeat; ++rep) {
      CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K,
                                &alpha,
                                dB, CUDA_R_16F, N,
                                dA, CUDA_R_16F, K,
                                &beta,
                                dC, CUDA_R_32F, N,
                                CUBLAS_COMPUTE_32F_FAST_16F,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
    ms /= kRepeat;

    // 拷贝结果回主机
    std::vector<float> ref(M * N);
    CHECK_CUDA(cudaMemcpy(ref.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // 校验
    bool ok = true;
    double max_abs_diff = 0.0;
    const char* check = "SKIP";
    if (do_cpu_verify) {
      ok = common::CheckEqual(cpu, ref, 1e-2f);
      max_abs_diff = common::MaxAbsDiff(cpu, ref);
      check = ok ? "PASS" : "FAIL";
    }

    // 计算并输出结果
    double gflops = (2.0 * M * N * K) / (ms * 1e6);
    std::cout << M << "x" << N << "x" << K << " | " << std::fixed << std::setprecision(4) << ms << " ms"
              << " | " << std::setprecision(1) << gflops << " GFLOP/s"
              << " | " << check << "\n";
    ofs << i << ",cublas_fp16," << M << "," << N << "," << K << ","
        << ms << "," << gflops << ","
        << max_abs_diff << "," << check << "\n";

    // 释放资源
    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
  }

  CHECK_CUBLAS(cublasDestroy(handle));
  return 0;
}
