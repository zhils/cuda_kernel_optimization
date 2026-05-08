/*
================================================================================
Flash Attention v0: Naive Baseline (Explicit Attention Matrix)
================================================================================
Implements the standard attention mechanism by materializing the full N×N
attention score matrix in global memory. While simple, this requires O(N²)
memory and bandwidth, making it impractical for long sequences.

Forward pass:
  1. S = Q @ K^T          [B, H, N, N]  ← attention scores
  2. P = softmax(S, dim=-1)               ← row-wise softmax
  3. O = P @ V            [B, H, N, D]   ← weighted sum of values

Key insight for optimization:
  The attention matrix S/P is the bottleneck — N² explodes quadratically.
  Flash Attention (v1) tiles the computation to never materialize S.
================================================================================
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

namespace attn_v0 {

constexpr int kWarpSize = 32;

// ----------------------------------------------------------------------------
// Kernel 1: Q @ K^T → S (attention scores)
// Each block handles one (b, h), computing scores[b,h,:,:] = Q[b,h,:,:] @ K[b,h,:,:]^T
// grid(B*H) × block(256) — 1 block per (batch, head)
// Uses 1D grid: S is (N, N) per head, tiled into 16-element chunks per thread
// ----------------------------------------------------------------------------
__global__ void ComputeScores(const float* __restrict__ Q,
                              const float* __restrict__ K,
                              float* __restrict__ S,
                              int B, int H, int N, int D) {
  const int bh = blockIdx.x;        // batch*H + head
  const int tid = threadIdx.x;       // 0..255
  const int total_threads = blockDim.x;

  // Each thread handles ceil(N/total_threads) rows of the N×N score matrix
  for (int row = tid; row < N; row += total_threads) {
    for (int col = 0; col < N; ++col) {
      float sum = 0.0f;
      // Q[bh, row, :] · K[bh, col, :]
      // Q layout: [b, h, n, d] -> idx = ((b*H + h) * N + row) * D + d
      // K layout: [b, h, n, d] -> idx = ((b*H + h) * N + col) * D + d
      const float* q_row = Q + ((static_cast<size_t>(bh) * N + row) * D);
      const float* k_row = K + ((static_cast<size_t>(bh) * N + col) * D);
      for (int d = 0; d < D; ++d) {
        sum += q_row[d] * k_row[d];
      }
      // S[bh, row, col] = Q[bh,row,:] · K[bh,col,:]
      S[(static_cast<size_t>(bh) * N + row) * N + col] = sum;
    }
  }
}

// ----------------------------------------------------------------------------
// Kernel 2: Row-wise softmax in-place on S, storing to P
// S[row, col] → P[row, col] = exp(S[row,col] - row_max) / row_sum
// ----------------------------------------------------------------------------
__global__ void SoftmaxScores(const float* __restrict__ S,
                              float* __restrict__ P,
                              int B, int H, int N) {
  const int bh = blockIdx.x;
  const int row = threadIdx.x;  // each thread handles one row
  if (row >= N) return;

  const float* s_row = S + (static_cast<size_t>(bh) * N + row) * N;
  float* p_row = P + (static_cast<size_t>(bh) * N + row) * N;

  // Find row max (for numerical stability)
  float max_val = -INFINITY;
  for (int col = 0; col < N; ++col) {
    max_val = fmaxf(max_val, s_row[col]);
  }

  // Compute exp and sum
  float sum = 0.0f;
  for (int col = 0; col < N; ++col) {
    float e = expf(s_row[col] - max_val);
    p_row[col] = e;
    sum += e;
  }

  // Normalize
  float inv_sum = 1.0f / sum;
  for (int col = 0; col < N; ++col) {
    p_row[col] *= inv_sum;
  }
}

// ----------------------------------------------------------------------------
// Kernel 3: P @ V → O
// O[bh, row, :] = sum_{col} P[bh, row, col] * V[bh, col, :]
// ----------------------------------------------------------------------------
__global__ void ApplyValues(const float* __restrict__ P,
                            const float* __restrict__ V,
                            float* __restrict__ O,
                            int B, int H, int N, int D) {
  const int bh = blockIdx.x;
  const int tid = threadIdx.x;
  const int total_threads = blockDim.x;

  for (int row = tid; row < N; row += total_threads) {
    for (int d = 0; d < D; ++d) {
      float sum = 0.0f;
      const float* p_row = P + (static_cast<size_t>(bh) * N + row) * N;
      for (int col = 0; col < N; ++col) {
        // P[bh, row, col] * V[bh, col, d]
        sum += p_row[col] * V[((static_cast<size_t>(bh) * N + col) * D) + d];
      }
      O[((static_cast<size_t>(bh) * N + row) * D) + d] = sum;
    }
  }
}

}  // namespace attn_v0

// ============================================================================
// CPU Reference (Full N² matrix, naive triple loop)
// ============================================================================
static void AttentionCPU(const float* Q, const float* K, const float* V,
                         float* O, int B, int H, int N, int D) {
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int row = 0; row < N; ++row) {
        // Compute S row = Q @ K^T
        float s_row[4096];  // assume N <= 4096
        float max_val = -INFINITY;
        for (int col = 0; col < N; ++col) {
          float sum = 0.0f;
          for (int d = 0; d < D; ++d) {
            sum += Q[((b*H + h)*N + row)*D + d] *
                   K[((b*H + h)*N + col)*D + d];
          }
          s_row[col] = sum;
          max_val = std::fmax(max_val, sum);
        }
        // softmax
        float row_sum = 0.0f;
        for (int col = 0; col < N; ++col) {
          s_row[col] = std::exp(s_row[col] - max_val);
          row_sum += s_row[col];
        }
        float inv_sum = 1.0f / row_sum;
        for (int col = 0; col < N; ++col) s_row[col] *= inv_sum;

        // P @ V
        for (int d = 0; d < D; ++d) {
          float sum = 0.0f;
          for (int col = 0; col < N; ++col) {
            sum += s_row[col] * V[((b*H + h)*N + col)*D + d];
          }
          O[((b*H + h)*N + row)*D + d] = sum;
        }
      }
    }
  }
}

// ============================================================================
// Main
// ============================================================================
int main() {
  constexpr int kWarmup = 1;
  constexpr int kRepeat = 10;

  // Test configurations: (B, H, N, D)
  std::vector<std::tuple<int, int, int, int>> test_cases = {
      {1, 1, 64, 32},
      {1, 1, 128, 64},
      {1, 2, 256, 64},
      {1, 4, 512, 64},
      {1, 8, 1024, 32},
  };

  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/flash_attention_v0_results.csv");
  ofs << "B,H,N,D,cpu_ms,gpu_ms,gflops,max_abs_diff,check\n";

  std::cout << "=== Flash Attention V0 (Naive, materialized N² matrix) ===\n";
  std::cout << std::left << std::setw(4) << "B" << std::setw(4) << "H"
            << std::setw(8) << "N" << std::setw(6) << "D"
            << std::setw(14) << "CPU ms" << std::setw(14) << "GPU ms"
            << std::setw(14) << "GFLOPS" << std::setw(9) << "Check" << "\n";
  std::cout << std::string(69, '-') << "\n";

  for (const auto& tc : test_cases) {
    int B = std::get<0>(tc), H = std::get<1>(tc), N = std::get<2>(tc), D = std::get<3>(tc);
    size_t total = static_cast<size_t>(B) * H * N * D;

    std::vector<float> Q(total), K(total), V(total), O_cpu(total), O_gpu(total);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& x : Q) x = dist(gen);
    for (auto& x : K) x = dist(gen);
    for (auto& x : V) x = dist(gen);

    // CPU reference
    auto t0 = std::chrono::high_resolution_clock::now();
    AttentionCPU(Q.data(), K.data(), V.data(), O_cpu.data(), B, H, N, D);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    float *dQ, *dK, *dV, *dS, *dP, *dO;
    size_t bh_count = static_cast<size_t>(B) * H;
    CHECK_CUDA(cudaMalloc(&dQ, total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dK, total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dV, total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dS, bh_count * N * N * sizeof(float)));  // O(N²)
    CHECK_CUDA(cudaMalloc(&dP, bh_count * N * N * sizeof(float)));  // O(N²)
    CHECK_CUDA(cudaMalloc(&dO, total * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dQ, Q.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, K.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, V.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid(bh_count);
    dim3 block_score(std::min(256, N));
    int block_softmax = (N > 256) ? N : 256;  // not great, but ok for demo

    for (int w = 0; w < kWarmup; ++w) {
      attn_v0::ComputeScores<<<grid, block_score>>>(dQ, dK, dS, B, H, N, D);
      attn_v0::SoftmaxScores<<<grid, block_softmax>>>(dS, dP, B, H, N);
      attn_v0::ApplyValues<<<grid, block_score>>>(dP, dV, dO, B, H, N, D);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    float gpu_ms = 0.0f;
    for (int rep = 0; rep < kRepeat; ++rep) {
      CHECK_CUDA(cudaEventRecord(s));
      attn_v0::ComputeScores<<<grid, block_score>>>(dQ, dK, dS, B, H, N, D);
      attn_v0::SoftmaxScores<<<grid, block_softmax>>>(dS, dP, B, H, N);
      attn_v0::ApplyValues<<<grid, block_score>>>(dP, dV, dO, B, H, N, D);
      CHECK_CUDA(cudaEventRecord(e));
      CHECK_CUDA(cudaEventSynchronize(e));
      float ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
      gpu_ms += ms;
    }
    gpu_ms /= kRepeat;

    CHECK_CUDA(cudaMemcpy(O_gpu.data(), dO, total * sizeof(float), cudaMemcpyDeviceToHost));

    double max_diff = common::MaxAbsDiff(O_cpu, O_gpu);
    bool ok = (max_diff < 1e-3);
    // FLOPs: 2*B*H*(N²*D + N²*D) = 4*B*H*N²*D (QK + PV matmuls, ignoring softmax)
    double gflops = 4.0 * B * H * N * N * D / (gpu_ms * 1e6);

    std::cout << std::left << std::setw(4) << B << std::setw(4) << H
              << std::setw(8) << N << std::setw(6) << D
              << std::fixed << std::setprecision(3) << std::setw(14) << cpu_ms
              << std::setw(14) << std::setprecision(4) << gpu_ms
              << std::setw(14) << std::setprecision(1) << gflops
              << std::setw(9) << (ok ? "PASS" : "FAIL") << "\n";

    ofs << B << "," << H << "," << N << "," << D << ","
        << cpu_ms << "," << gpu_ms << "," << gflops << "," << max_diff
        << "," << (ok ? "PASS" : "FAIL") << "\n";

    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(dQ)); CHECK_CUDA(cudaFree(dK)); CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dS)); CHECK_CUDA(cudaFree(dP)); CHECK_CUDA(cudaFree(dO));
  }

  std::cout << "\nResults saved to data/results/flash_attention_v0_results.csv\n";
  return 0;
}
