#include <cuda_runtime.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

__global__ void GemmGpuOptKernel(const float* A, const float* B, float* C, int M, int N,
                                 int K) {
  constexpr int kTile = 16;

  // +1 padding to reduce shared-memory bank conflict.
  __shared__ float As[kTile][kTile + 1];
  __shared__ float Bs[kTile][kTile + 1];

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

    #pragma unroll
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
  auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/gemm2_naive_results.csv");
  ofs << "id,M,N,K,cpu_ms,gpu_ms,speedup,max_abs_diff,check\n";
  for (size_t i = 0; i < cases.size(); ++i) {
    int M = cases[i].rows, N = cases[i].cols, K = (M + N) / 2;
    std::vector<float> A(static_cast<size_t>(M) * K), B(static_cast<size_t>(K) * N),
        cpu(static_cast<size_t>(M) * N), gpu(static_cast<size_t>(M) * N);
    common::InitMatrix(A, M, K);
    common::InitMatrix(B, K, N);

    double cpu_ms_sum = 0.0;
    for (int rep = 0; rep < kRepeat; ++rep) {
      auto t0 = std::chrono::high_resolution_clock::now();
      GemmCPU(A.data(), B.data(), cpu.data(), M, N, K);
      auto t1 = std::chrono::high_resolution_clock::now();
      cpu_ms_sum += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double cpu_ms = cpu_ms_sum / kRepeat;

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
    // Warmup once before averaged timing.
    GemmGpuOptKernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaEventRecord(s));
    for (int rep = 0; rep < kRepeat; ++rep) {
      GemmGpuOptKernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    CHECK_CUDA(cudaGetLastError());
    float gpu_ms_total = 0;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_total, s, e));
    const float gpu_ms = gpu_ms_total / kRepeat;
    CHECK_CUDA(cudaMemcpy(gpu.data(), dC, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = common::CheckEqual(cpu, gpu, 1e-3f);
    ofs << i << "," << M << "," << N << "," << K << "," << cpu_ms << "," << gpu_ms << ","
        << (gpu_ms > 0 ? cpu_ms / gpu_ms : 0) << "," << common::MaxAbsDiff(cpu, gpu) << ","
        << (ok ? "PASS" : "FAIL") << "\n";

    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
  }
  return 0;
}
