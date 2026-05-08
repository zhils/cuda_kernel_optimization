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

namespace attn_v1 {
constexpr int kBr = 32;
constexpr int kBc = 32;
constexpr int kBlockSize = 256;

// Each thread maintains O_accum for ALL Q rows in its warp, across ALL KV tiles.
// Key insight: O stays in register, only written to global once after all tiles.
__global__ void FlashAttentionKernel(const float* __restrict__ Q,
                                     const float* __restrict__ K,
                                     const float* __restrict__ V,
                                     float* __restrict__ O,
                                     int B, int H, int N, int D) {
  extern __shared__ float shared_mem[];
  float* Q_smem = shared_mem;
  float* K_smem = shared_mem + kBr * D;
  float* V_smem = shared_mem + kBr * D + kBc * D;

  const int bh = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane = tid % 32;
  const int num_q_tiles = (N + kBr - 1) / kBr;
  const int num_kv_tiles = (N + kBc - 1) / kBc;
  const int rows_per_warp = kBr / (kBlockSize / 32);  // 32 / 8 = 4

  for (int q_tile = 0; q_tile < num_q_tiles; ++q_tile) {
    const int q_start = q_tile * kBr;
    const int q_size = min(kBr, N - q_start);

    for (int idx = tid; idx < kBr * D; idx += kBlockSize) {
      const int r = idx / D, d = idx % D;
      Q_smem[r * D + d] = (q_start + r < N)
          ? Q[((size_t(bh) * N + q_start + r) * D) + d] : 0.0f;
    }
    __syncthreads();

    const int my_row_start = warp_id * rows_per_warp;
    const int my_row_end = min(my_row_start + rows_per_warp, q_size);

    // Register accumulators: each thread holds O[d] for its assigned D elements
    // We need separate accumulators for each Q row this warp handles
    // rows_per_warp copies, each with ceil(D/32) floats per lane
    constexpr int kMaxDPerLane = 4;  // D <= 128
    float o_accum[4][kMaxDPerLane] = {{0.0f}};
    float m_prev[4], ell_prev[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) { m_prev[i] = -INFINITY; ell_prev[i] = 0.0f; }

    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
      const int kv_start = kv_tile * kBc;
      const int kv_size = min(kBc, N - kv_start);

      for (int idx = tid; idx < kBc * D; idx += kBlockSize) {
        const int r = idx / D, d = idx % D;
        K_smem[r * D + d] = (kv_start + r < N)
            ? K[((size_t(bh) * N + kv_start + r) * D) + d] : 0.0f;
        V_smem[r * D + d] = (kv_start + r < N)
            ? V[((size_t(bh) * N + kv_start + r) * D) + d] : 0.0f;
      }
      __syncthreads();

      for (int ri = 0; ri < rows_per_warp; ++ri) {
        const int q_row = my_row_start + ri;
        if (q_row >= q_size) continue;

        // S_row = Q_row @ K_tile^T [1 × kBc]
        float S_row[32];
        for (int c = 0; c < kv_size; ++c) {
          float sum = 0.0f;
          for (int d = lane; d < D; d += 32)
            sum += Q_smem[q_row * D + d] * K_smem[c * D + d];
          for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
          S_row[c] = __shfl_sync(0xffffffff, sum, 0);
        }

        // Online softmax rescaling
        float row_max = -INFINITY;
        for (int c = 0; c < kv_size; ++c)
          row_max = fmaxf(row_max, S_row[c]);
        float m_new = fmaxf(m_prev[ri], row_max);
        float old_scale = expf(m_prev[ri] - m_new);

        // O_accum[d] = O_accum[d] * old_scale + sum_c(exp(S[c]-m_new) * V[c,d])
        for (int d = lane; d < D; d += 32) {
          float new_contrib = 0.0f;
          for (int c = 0; c < kv_size; ++c)
            new_contrib += expf(S_row[c] - m_new) * V_smem[c * D + d];
          o_accum[ri][d / 32] = o_accum[ri][d / 32] * old_scale + new_contrib;
        }

        // Update ℓ (compute on lane 0, broadcast)
        if (lane == 0) {
          float new_sum = 0.0f;
          for (int c = 0; c < kv_size; ++c)
            new_sum += expf(S_row[c] - m_new);
          ell_prev[ri] = ell_prev[ri] * old_scale + new_sum;
        }
        m_prev[ri] = m_new;
      }
      __syncthreads();
    }

    // Normalize and write: O[d] /= ℓ for each row
    for (int ri = 0; ri < rows_per_warp; ++ri) {
      const int q_row = my_row_start + ri;
      if (q_row >= q_size) continue;
      float inv_ell = __shfl_sync(0xffffffff, 1.0f / ell_prev[ri], 0);
      const size_t base = (size_t(bh) * N + q_start + q_row) * D;
      for (int d = lane; d < D; d += 32)
        O[base + d] = o_accum[ri][d / 32] * inv_ell;
    }
    __syncthreads();
  }
}
}  // namespace attn_v1

static void AttentionCPU(const float* Q, const float* K, const float* V,
                         float* O, int B, int H, int N, int D) {
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int row = 0; row < N; ++row) {
        float s_row[4096], max_val = -INFINITY;
        for (int col = 0; col < N; ++col) {
          float sum = 0.0f;
          for (int d = 0; d < D; ++d)
            sum += Q[((b*H+h)*N+row)*D+d] * K[((b*H+h)*N+col)*D+d];
          s_row[col] = sum; max_val = fmaxf(max_val, sum);
        }
        float row_sum = 0.0f;
        for (int col = 0; col < N; ++col) { s_row[col] = expf(s_row[col]-max_val); row_sum += s_row[col]; }
        float inv = 1.0f / row_sum;
        for (int col = 0; col < N; ++col) s_row[col] *= inv;
        for (int d = 0; d < D; ++d) {
          float sum = 0.0f;
          for (int col = 0; col < N; ++col) sum += s_row[col] * V[((b*H+h)*N+col)*D+d];
          O[((b*H+h)*N+row)*D+d] = sum;
        }
      }
}

int main() {
  constexpr int kWarmup = 1, kRepeat = 10;
  std::vector<std::tuple<int,int,int,int>> cases = {{1,1,64,32},{1,1,128,64},{1,2,256,64},{1,4,512,64},{1,8,1024,32}};
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/flash_attention_v1_results.csv");
  ofs << "B,H,N,D,cpu_ms,gpu_ms,gflops,max_abs_diff,check\n";
  std::cout << "=== Flash Attention V1 (Correct Online Softmax) ===\n"
            << "B   H   N       D     CPU ms        GPU ms        GFLOPS  Check\n"
            << std::string(64,'-') << "\n";

  for (auto [B, H, N, D] : cases) {
    size_t total = size_t(B)*H*N*D;
    std::vector<float> Q(total), K(total), V(total), O_cpu(total), O_gpu(total);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& x : Q) x = dist(gen);
    for (auto& x : K) x = dist(gen);
    for (auto& x : V) x = dist(gen);

    auto t0 = std::chrono::high_resolution_clock::now();
    AttentionCPU(Q.data(), K.data(), V.data(), O_cpu.data(), B, H, N, D);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double,std::milli>(t1-t0).count();

    float *dQ, *dK, *dV, *dO;
    CHECK_CUDA(cudaMalloc(&dQ, total*4)); CHECK_CUDA(cudaMalloc(&dK, total*4));
    CHECK_CUDA(cudaMalloc(&dV, total*4)); CHECK_CUDA(cudaMalloc(&dO, total*4));
    CHECK_CUDA(cudaMemcpy(dQ, Q.data(), total*4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, K.data(), total*4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, V.data(), total*4, cudaMemcpyHostToDevice));

    size_t smem = (attn_v1::kBr + 2*attn_v1::kBc)*D*sizeof(float);
    dim3 grid(B*H), block(attn_v1::kBlockSize);
    CHECK_CUDA(cudaFuncSetAttribute(attn_v1::FlashAttentionKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    for (int w = 0; w < kWarmup; ++w)
      attn_v1::FlashAttentionKernel<<<grid,block,smem>>>(dQ,dK,dV,dO,B,H,N,D);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t s,e;
    CHECK_CUDA(cudaEventCreate(&s)); CHECK_CUDA(cudaEventCreate(&e));
    float gpu_ms = 0.0f;
    for (int rep = 0; rep < kRepeat; ++rep) {
      CHECK_CUDA(cudaEventRecord(s));
      attn_v1::FlashAttentionKernel<<<grid,block,smem>>>(dQ,dK,dV,dO,B,H,N,D);
      CHECK_CUDA(cudaEventRecord(e));
      CHECK_CUDA(cudaEventSynchronize(e));
      float ms; CHECK_CUDA(cudaEventElapsedTime(&ms,s,e)); gpu_ms += ms;
    }
    gpu_ms /= kRepeat;
    CHECK_CUDA(cudaMemcpy(O_gpu.data(), dO, total*4, cudaMemcpyDeviceToHost));

    double diff = common::MaxAbsDiff(O_cpu, O_gpu);
    bool ok = diff < 1e-3;
    double gflops = 4.0*B*H*N*N*D/(gpu_ms*1e6);
    std::cout << B << " " << H << "   " << N << "      " << D << "     "
              << std::fixed << std::setprecision(3) << cpu_ms << "       "
              << gpu_ms << "       " << std::setprecision(1) << gflops << "     "
              << (ok?"PASS":"FAIL") << "\n";
    ofs << B<<","<<H<<","<<N<<","<<D<<","<<cpu_ms<<","<<gpu_ms<<","<<gflops<<","<<diff<<","<<(ok?"PASS":"FAIL")<<"\n";
    CHECK_CUDA(cudaEventDestroy(s)); CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(dQ)); CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV)); CHECK_CUDA(cudaFree(dO));
  }
  std::cout << "\nSaved to data/results/flash_attention_v1_results.csv\n";
}
