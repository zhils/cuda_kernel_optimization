#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"
// inline CpuGemm defined below

static void GemmCPU(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double sum = 0;
      for (int k = 0; k < K; ++k) sum += static_cast<double>(A[i * K + k]) * B[k * N + j];
      C[i * N + j] = static_cast<float>(sum);
    }
}

__global__ void GemmNaiveKernel(const float* A, const float* B, float* C, int M, int N, int K) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float sum = 0.f;
    for (int k = 0; k < K; ++k) sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
  }
}

int main() {
  // 参数与输出文件准备
  constexpr int kWarmup = 1;
  constexpr int kRepeat = 10;
  constexpr int kMaxCpuVerifyDim = 1024;
  constexpr int kMaxGpuRunDim = 8192;
  auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/gemm_naive_results.csv");
  ofs << "id,group,M,N,K,cpu_ms,gpu_ms,speedup,max_abs_diff,check\n";

  std::cout << "=== GEMM V0 (Naive) ===\n";

  for (size_t i = 0; i < cases.size(); ++i) {
    int M = cases[i].rows, N = cases[i].cols, K = M;
    const bool do_gpu_run = (M <= kMaxGpuRunDim && N <= kMaxGpuRunDim && K <= kMaxGpuRunDim);

    // 生成测试数据
    std::vector<float> A(M * K), B(K * N), cpu(M * N), gpu(M * N);
    common::InitMatrix(A, M, K);
    common::InitMatrix(B, K, N);

    // CPU参考计算
    const bool do_cpu_verify = (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim && K <= kMaxCpuVerifyDim);
    double cpu_ms = 0.0;
    if (do_cpu_verify) {
      auto t0 = std::chrono::high_resolution_clock::now();
      GemmCPU(A.data(), B.data(), cpu.data(), M, N, K);
      auto t1 = std::chrono::high_resolution_clock::now();
      cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    float gpu_ms = 0.0f;
    bool ok = true;
    double max_abs_diff = 0.0;
    const char* check = "SKIP";

    if (do_gpu_run) {
      // 分配GPU内存
      float *dA, *dB, *dC;
      CHECK_CUDA(cudaMalloc(&dA, A.size() * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&dB, B.size() * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&dC, cpu.size() * sizeof(float)));

      // 拷贝数据到GPU
      CHECK_CUDA(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

      // 启动配置与预热
      dim3 block(16, 16), grid((N + 15) / 16, (M + 15) / 16);
      for (int w = 0; w < kWarmup; ++w) {
        GemmNaiveKernel<<<grid, block>>>(dA, dB, dC, M, N, K);
      }

      // 计时循环
      cudaEvent_t s, e;
      CHECK_CUDA(cudaEventCreate(&s));
      CHECK_CUDA(cudaEventCreate(&e));
      std::vector<float> gpu_times;
      gpu_times.reserve(kRepeat);
      for (int rep = 0; rep < kRepeat; ++rep) {
        CHECK_CUDA(cudaEventRecord(s));
        GemmNaiveKernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));
        CHECK_CUDA(cudaGetLastError());
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
        gpu_times.push_back(ms);
      }
      std::sort(gpu_times.begin(), gpu_times.end());
      if (gpu_times.size() > 2) {
        for (size_t t = 1; t + 1 < gpu_times.size(); ++t) gpu_ms += gpu_times[t];
        gpu_ms /= static_cast<float>(gpu_times.size() - 2);
      } else if (!gpu_times.empty()) {
        for (float t : gpu_times) gpu_ms += t;
        gpu_ms /= static_cast<float>(gpu_times.size());
      }

      // 拷贝结果回主机
      CHECK_CUDA(cudaMemcpy(gpu.data(), dC, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

      // 释放GPU资源
      CHECK_CUDA(cudaEventDestroy(s));
      CHECK_CUDA(cudaEventDestroy(e));
      CHECK_CUDA(cudaFree(dA));
      CHECK_CUDA(cudaFree(dB));
      CHECK_CUDA(cudaFree(dC));

      // 校验
      if (do_cpu_verify) {
        ok = common::CheckEqual(cpu, gpu, 1e-3f);
        max_abs_diff = common::MaxAbsDiff(cpu, gpu);
        check = ok ? "PASS" : "FAIL";
      }
    } else {
      check = "SKIP_GPU_LARGE";
    }

    // 计算并输出结果
    double gflops = (2.0 * M * N * K) / (gpu_ms * 1e6);
    std::cout << M << "x" << N << "x" << K << " | " << std::fixed << std::setprecision(4) << gpu_ms << " ms"
              << " | " << std::setprecision(1) << gflops << " GFLOP/s"
              << " | " << check << "\n";

    ofs << cases[i].id << "," << cases[i].group << "," << M << "," << N << "," << K << "," << cpu_ms << "," << gpu_ms
        << "," << (gpu_ms > 0 ? cpu_ms / gpu_ms : 0) << "," << max_abs_diff << "," << check
        << "\n";
  }
  return 0;
}
