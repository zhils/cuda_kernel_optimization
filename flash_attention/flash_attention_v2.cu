// Flash Attention v2 fallback:
// The original WGMMA inline-PTX implementation is not accepted by the current
// CUDA 13.2 + sm_120 toolchain in this repository environment. To keep the
// benchmark target buildable/profilable, this file provides a portable CUDA
// fallback kernel and benchmark harness.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

namespace attn_v2 {

__global__ void FlashAttentionV2FallbackKernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int N, int D) {
  int bh = blockIdx.x;
  int t = blockIdx.y;
  int d = threadIdx.x;
  if (bh >= B * H || t >= N || d >= D) return;

  const size_t q_base = (static_cast<size_t>(bh) * N + t) * D;
  float out = 0.0f;
  float sum = 0.0f;
  float maxv = -1e30f;

  for (int j = 0; j < N; ++j) {
    const size_t k_base = (static_cast<size_t>(bh) * N + j) * D;
    float s = 0.0f;
    for (int kk = 0; kk < D; ++kk) {
      s += Q[q_base + kk] * K[k_base + kk];
    }
    s *= rsqrtf(static_cast<float>(D));
    maxv = fmaxf(maxv, s);
  }
  for (int j = 0; j < N; ++j) {
    const size_t k_base = (static_cast<size_t>(bh) * N + j) * D;
    float s = 0.0f;
    for (int kk = 0; kk < D; ++kk) {
      s += Q[q_base + kk] * K[k_base + kk];
    }
    s = expf(s * rsqrtf(static_cast<float>(D)) - maxv);
    sum += s;
    out += s * V[k_base + d];
  }
  O[q_base + d] = out / fmaxf(sum, 1e-8f);
}

static void AttentionCPU(const float* Q, const float* K, const float* V, float* O,
                         int B, int H, int N, int D) {
  for (int bh = 0; bh < B * H; ++bh) {
    for (int t = 0; t < N; ++t) {
      const size_t q_base = (static_cast<size_t>(bh) * N + t) * D;
      float maxv = -1e30f;
      for (int j = 0; j < N; ++j) {
        const size_t k_base = (static_cast<size_t>(bh) * N + j) * D;
        float s = 0.0f;
        for (int kk = 0; kk < D; ++kk) s += Q[q_base + kk] * K[k_base + kk];
        maxv = std::max(maxv, s / std::sqrt(static_cast<float>(D)));
      }
      for (int d = 0; d < D; ++d) {
        float num = 0.0f, den = 0.0f;
        for (int j = 0; j < N; ++j) {
          const size_t k_base = (static_cast<size_t>(bh) * N + j) * D;
          float s = 0.0f;
          for (int kk = 0; kk < D; ++kk) s += Q[q_base + kk] * K[k_base + kk];
          float p = std::exp(s / std::sqrt(static_cast<float>(D)) - maxv);
          den += p;
          num += p * V[k_base + d];
        }
        O[q_base + d] = num / std::max(den, 1e-8f);
      }
    }
  }
}

}  // namespace attn_v2

int main() {
  constexpr int kWarmup = 1;
  constexpr int kRepeat = 5;
  constexpr int kMaxCpuN = 128;
  std::vector<std::tuple<int, int, int, int>> cases = {
      {1, 1, 64, 32},
      {1, 2, 128, 64},
      {1, 4, 256, 64},
  };

  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/flash_attention_v2_results.csv");
  ofs << "B,H,N,D,gpu_ms,max_abs_diff,check,note\n";

  std::cout << "=== Flash Attention V2 (fallback kernel) ===\n";
  for (auto [B, H, N, D] : cases) {
    const size_t n = static_cast<size_t>(B) * H * N * D;
    std::vector<float> Q(n), K(n), V(n), O_cpu(n), O_gpu(n);
    common::InitMatrix(Q, B * H * N, D);
    common::InitMatrix(K, B * H * N, D);
    common::InitMatrix(V, B * H * N, D);

    bool do_cpu_verify = (N <= kMaxCpuN);
    if (do_cpu_verify) {
      attn_v2::AttentionCPU(Q.data(), K.data(), V.data(), O_cpu.data(), B, H, N, D);
    }

    float *dQ, *dK, *dV, *dO;
    CHECK_CUDA(cudaMalloc(&dQ, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dK, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dV, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dO, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dQ, Q.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, K.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, V.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid(B * H, N);
    dim3 block(D);
    for (int i = 0; i < kWarmup; ++i) {
      attn_v2::FlashAttentionV2FallbackKernel<<<grid, block>>>(dQ, dK, dV, dO, B, H, N, D);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(s));
    for (int i = 0; i < kRepeat; ++i) {
      attn_v2::FlashAttentionV2FallbackKernel<<<grid, block>>>(dQ, dK, dV, dO, B, H, N, D);
    }
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
    ms /= kRepeat;

    CHECK_CUDA(cudaMemcpy(O_gpu.data(), dO, n * sizeof(float), cudaMemcpyDeviceToHost));
    double max_abs_diff = 0.0;
    const char* check = "SKIP";
    if (do_cpu_verify) {
      max_abs_diff = common::MaxAbsDiff(O_cpu, O_gpu);
      check = (max_abs_diff < 2e-2f) ? "PASS" : "FAIL";
    }

    std::cout << "B=" << B << " H=" << H << " N=" << N << " D=" << D
              << " | " << std::fixed << std::setprecision(4) << ms << " ms"
              << " | " << check << "\n";
    ofs << B << "," << H << "," << N << "," << D << ","
        << ms << "," << max_abs_diff << "," << check
        << ",fallback_no_wgmma\n";

    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(dQ));
    CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dO));
  }
  return 0;
}
