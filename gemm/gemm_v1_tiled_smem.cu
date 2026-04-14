#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

__global__ void GemmOptimizedKernel(const float* A, const float* B, float* C, int M, int N,
                                    int K) {
  constexpr int kTile = 16;

  __shared__ float As[kTile][kTile];
  __shared__ float Bs[kTile][kTile];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * kTile + ty;
  const int col = blockIdx.x * kTile + tx;
  float sum = 0.0f;

  for (int k0 = 0; k0 < K; k0 += kTile) {
    const int a_col = k0 + tx;
    const int b_row = k0 + ty;

    As[ty][tx] = (row < M && a_col < K) ? __ldg(A + row * K + a_col) : 0.0f;
    Bs[ty][tx] = (b_row < K && col < N) ? __ldg(B + b_row * N + col) : 0.0f;

    __syncthreads();

    for (int kk = 0; kk < kTile; ++kk) {
      sum += As[ty][kk] * Bs[kk][tx];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

static void GemmCPU(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < N; ++c) {
      float s = 0;
      for (int k = 0; k < K; ++k) s += A[r * K + k] * B[k * N + c];
      C[r * N + c] = s;
    }
}

int main() {
  constexpr int kRepeat = 10;
  constexpr int kMaxCpuVerifyDim = 1024;
  constexpr int kMaxGpuRunDim = 2048;
  auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/gemm1_naive_results.csv");
  ofs << "id,M,N,K,cpu_ms,gpu_ms,speedup,max_abs_diff,check\n";
  for (size_t i = 0; i < cases.size(); ++i) {
    int M = cases[i].rows, N = cases[i].cols, K = (M + N) / 2;
    const bool do_gpu_run = (M <= kMaxGpuRunDim && N <= kMaxGpuRunDim && K <= kMaxGpuRunDim);
    std::vector<float> A(M * K), B(K * N), cpu(M * N), gpu(M * N);
    common::InitMatrix(A, M, K);
    common::InitMatrix(B, K, N);
    const bool do_cpu_verify = (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim && K <= kMaxCpuVerifyDim);
    double cpu_ms_sum = 0.0;
    if (do_cpu_verify) {
      for (int rep = 0; rep < kRepeat; ++rep) {
        auto t0 = std::chrono::high_resolution_clock::now();
        GemmCPU(A.data(), B.data(), cpu.data(), M, N, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_ms_sum += std::chrono::duration<double, std::milli>(t1 - t0).count();
      }
    }
    double cpu_ms = cpu_ms_sum / kRepeat;
    float gpu_ms = 0.0f;
    if (do_gpu_run) {
      float *dA, *dB, *dC;
      CHECK_CUDA(cudaMalloc(&dA, A.size() * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&dB, B.size() * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&dC, cpu.size() * sizeof(float)));
      CHECK_CUDA(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));
      constexpr int kTile = 16;
      dim3 block(kTile, kTile), grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
      cudaEvent_t s, e;
      CHECK_CUDA(cudaEventCreate(&s));
      CHECK_CUDA(cudaEventCreate(&e));
      GemmOptimizedKernel<<<grid, block>>>(dA, dB, dC, M, N, K);
      std::vector<float> gpu_times;
      gpu_times.reserve(kRepeat);
      for (int rep = 0; rep < kRepeat; ++rep) {
        CHECK_CUDA(cudaEventRecord(s));
        GemmOptimizedKernel<<<grid, block>>>(dA, dB, dC, M, N, K);
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
      CHECK_CUDA(cudaMemcpy(gpu.data(), dC, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaEventDestroy(s));
      CHECK_CUDA(cudaEventDestroy(e));
      CHECK_CUDA(cudaFree(dA));
      CHECK_CUDA(cudaFree(dB));
      CHECK_CUDA(cudaFree(dC));
    }
    bool ok = true;
    double max_abs_diff = 0.0;
    const char* check = "SKIP";
    if (!do_gpu_run) {
      check = "SKIP_GPU_LARGE";
    } else if (do_cpu_verify) {
      ok = common::CheckEqual(cpu, gpu, 1e-3f);
      max_abs_diff = common::MaxAbsDiff(cpu, gpu);
      check = ok ? "PASS" : "FAIL";
    }
    ofs << i << "," << M << "," << N << "," << K << "," << cpu_ms << "," << gpu_ms
        << "," << (gpu_ms > 0 ? cpu_ms / gpu_ms : 0) << "," << max_abs_diff << "," << check
        << "\n";
  }
  return 0;
}
