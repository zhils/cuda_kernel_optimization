/*
================================================================================
Flash Attention v0：朴素基线（显示注意力矩阵）
================================================================================
通过在全局内存中实例化完整的 N×N 注意力分数矩阵，实现标准注意力机制。
虽然简单，但需要 O(N²) 的内存和带宽，使得长序列情况下不切实际。

前向传播：
  1. S = Q @ K^T          [B, H, N, N]  ← 注意力分数
  2. P = softmax(S, dim=-1)               ← 行-wise softmax
  3. O = P @ V            [B, H, N, D]   ← 值加权求和

优化关键洞察：
  注意力矩阵 S/P 是瓶颈 — N² 以二次方速度爆炸。
  Flash Attention (v1) 通过将计算分块，从不实例化 S。
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
// Kernel 1: Q @ K^T → S（注意力分数）
// 每个 block 处理一个 (b, h)，计算 scores[b,h,:,:] = Q[b,h,:,:] @ K[b,h,:,:]^T
// grid(B*H) × block(256) — 每个 (batch, head) 一个 block
// 使用 1D grid：每个 head 的 S 是 (N, N)，每个线程分块处理 16 个元素
// ----------------------------------------------------------------------------
__global__ void ComputeScores(const float* __restrict__ Q,
                              const float* __restrict__ K,
                              float* __restrict__ S,
                              int B, int H, int N, int D) {
  const int bh = blockIdx.x;
  const int tid = threadIdx.x;
  const int total_threads = blockDim.x;

  for (int row = tid; row < N; row += total_threads) {
    for (int col = 0; col < N; ++col) {
      float sum = 0.0f;
      const float* q_row = Q + ((static_cast<size_t>(bh) * N + row) * D);
      const float* k_row = K + ((static_cast<size_t>(bh) * N + col) * D);
      for (int d = 0; d < D; ++d) {
        sum += q_row[d] * k_row[d];
      }
      S[(static_cast<size_t>(bh) * N + row) * N + col] = sum;
    }
  }
}

// ----------------------------------------------------------------------------
// Kernel 2: 行-wise softmax（原地对 S 操作，结果存到 P）
// S[row, col] → P[row, col] = exp(S[row,col] - row_max) / row_sum
// ----------------------------------------------------------------------------
__global__ void SoftmaxScores(const float* __restrict__ S,
                              float* __restrict__ P,
                              int B, int H, int N) {
  const int bh = blockIdx.x;
  const int row = threadIdx.x;
  if (row >= N) return;

  const float* s_row = S + (static_cast<size_t>(bh) * N + row) * N;
  float* p_row = P + (static_cast<size_t>(bh) * N + row) * N;

  float max_val = -INFINITY;
  for (int col = 0; col < N; ++col) {
    max_val = fmaxf(max_val, s_row[col]);
  }

  float sum = 0.0f;
  for (int col = 0; col < N; ++col) {
    float e = expf(s_row[col] - max_val);
    p_row[col] = e;
    sum += e;
  }

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
        sum += p_row[col] * V[((static_cast<size_t>(bh) * N + col) * D) + d];
      }
      O[((static_cast<size_t>(bh) * N + row) * D) + d] = sum;
    }
  }
}

}  // namespace attn_v0

// ============================================================================
// CPU 参考实现（完整 N² 矩阵，朴素三重循环）
// ============================================================================
static void AttentionCPU(const float* Q, const float* K, const float* V,
                         float* O, int B, int H, int N, int D) {
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int row = 0; row < N; ++row) {
        float s_row[4096];
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
        float row_sum = 0.0f;
        for (int col = 0; col < N; ++col) {
          s_row[col] = std::exp(s_row[col] - max_val);
          row_sum += s_row[col];
        }
        float inv_sum = 1.0f / row_sum;
        for (int col = 0; col < N; ++col) s_row[col] *= inv_sum;

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
// 主函数
// ============================================================================
int main() {
  constexpr int kWarmup = 1;
  constexpr int kRepeat = 10;

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

  std::cout << "=== Flash Attention V0 (朴素, 实例化 N² 矩阵) ===\n";
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

    auto t0 = std::chrono::high_resolution_clock::now();
    AttentionCPU(Q.data(), K.data(), V.data(), O_cpu.data(), B, H, N, D);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    float *dQ, *dK, *dV, *dS, *dP, *dO;
    size_t bh_count = static_cast<size_t>(B) * H;
    CHECK_CUDA(cudaMalloc(&dQ, total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dK, total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dV, total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dS, bh_count * N * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dP, bh_count * N * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dO, total * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dQ, Q.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, K.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, V.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid(bh_count);
    dim3 block_score(std::min(256, N));
    int block_softmax = (N > 256) ? N : 256;

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

  std::cout << "\n结果已保存到 data/results/flash_attention_v0_results.csv\n";
  return 0;
}
