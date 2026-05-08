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

namespace fused_v2 {

// ============================================================================
// Fused Conv1D + SiLU v2: Full-parallelism dual-kernel fusion
// ============================================================================
// Fuses 5 operations into 2 kernels (from v0's 5 kernels):
//   Kernel A (ComputQKVZ):  linear_qkv + linear_z + split_qkv
//   Kernel B (ConvGate):    causal_conv1d + SiLU + gate_mul
//
// Key differences from v1:
// - Thread mapping: one thread per (b,t,h) → B×L×H threads, matching v0
// - Restores full GPU parallelism vs v1's serial-L bottleneck
// - 1 intermediate buffer (z_proj) vs v0's 7 intermediate buffers
// - float4 vectorized global memory loads
// ============================================================================

constexpr int kBlockSize = 256;

// ----------------------------------------------------------------------------
// Kernel A: Compute Q, K, V_raw, z_proj from x
// Thread mapping: grid(B*L, ceil(H/256)) × block(256)
// ----------------------------------------------------------------------------
__global__ void ComputeQKVZKernel(const float* __restrict__ x,
                                  const float* __restrict__ W_qkv,
                                  const float* __restrict__ b_qkv,
                                  const float* __restrict__ W_z,
                                  const float* __restrict__ b_z,
                                  float* __restrict__ Q,
                                  float* __restrict__ K,
                                  float* __restrict__ V,
                                  float* __restrict__ z_proj,
                                  int B, int L, int D, int H) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_threads = blockDim.x * gridDim.x;
  for (int idx = global_idx; idx < B * L * H; idx += total_threads) {
    int b = idx / (L * H);
    int rem1 = idx % (L * H);
    int t = rem1 / H;
    int h = rem1 % H;

    const float* x_bt = x + (static_cast<size_t>(b) * L + t) * D;
    const float* Wq = W_qkv + static_cast<size_t>(h) * D;
    const float* Wk = W_qkv + static_cast<size_t>(h + H) * D;
    const float* Wv = W_qkv + static_cast<size_t>(h + 2 * H) * D;
    const float* Wz = W_z + static_cast<size_t>(h) * D;

    float q_raw = b_qkv[h];
    float k_raw = b_qkv[h + H];
    float v_raw = b_qkv[h + 2 * H];
    float zp = b_z[h];

    int d = 0;
    for (; d + 3 < D; d += 4) {
      const float4 xv = reinterpret_cast<const float4*>(x_bt)[d / 4];
      const float4 wq = reinterpret_cast<const float4*>(Wq)[d / 4];
      const float4 wk = reinterpret_cast<const float4*>(Wk)[d / 4];
      const float4 wv = reinterpret_cast<const float4*>(Wv)[d / 4];
      const float4 wz = reinterpret_cast<const float4*>(Wz)[d / 4];
      q_raw += xv.x * wq.x + xv.y * wq.y + xv.z * wq.z + xv.w * wq.w;
      k_raw += xv.x * wk.x + xv.y * wk.y + xv.z * wk.z + xv.w * wk.w;
      v_raw += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
      zp    += xv.x * wz.x + xv.y * wz.y + xv.z * wz.z + xv.w * wz.w;
    }
    for (; d < D; ++d) {
      const float xv = x_bt[d];
      q_raw += xv * Wq[d];
      k_raw += xv * Wk[d];
      v_raw += xv * Wv[d];
      zp    += xv * Wz[d];
    }

    const size_t out_idx = (static_cast<size_t>(b) * L + t) * H + h;
    Q[out_idx] = q_raw;
    K[out_idx] = k_raw;
    V[out_idx] = v_raw;
    z_proj[out_idx] = zp;
  }
}

// ----------------------------------------------------------------------------
// Kernel B: Compute V = V_raw * SiLU(CausalConv1d(z_proj))
// Thread mapping: grid(B*L, ceil(H/256)) × block(256)
// ----------------------------------------------------------------------------
__global__ void ConvGateKernel(const float* __restrict__ z_proj,
                               const float* __restrict__ K_conv,
                               float* __restrict__ V,
                               int B, int L, int H, int k_size) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_threads = blockDim.x * gridDim.x;
  for (int idx = global_idx; idx < B * L * H; idx += total_threads) {
    int b = idx / (L * H);
    int rem1 = idx % (L * H);
    int t = rem1 / H;
    int h = rem1 % H;

    const size_t base = (static_cast<size_t>(b) * L) * H;

    float z_conv = 0.0f;
    for (int i = 0; i < k_size; ++i) {
      int ti = t - i;
      if (ti < 0) continue;
      z_conv += z_proj[base + static_cast<size_t>(ti) * H + h] * K_conv[static_cast<size_t>(i) * H + h];
    }

    const float sigmoid = 1.0f / (1.0f + expf(-z_conv));
    const float z_act = z_conv * sigmoid;

    const size_t out_idx = base + static_cast<size_t>(t) * H + h;
    V[out_idx] *= z_act;
  }
}

}  // namespace fused_v2

// ============================================================================
// CPU Reference Implementation
// ============================================================================
static void FusedConv1dSiLU_CPU(const float* x,
                                const float* W_qkv, const float* b_qkv,
                                const float* W_z, const float* b_z,
                                const float* K_conv,
                                float* Q, float* K, float* V,
                                int B, int L, int D, int H, int k_size) {
  for (int ib = 0; ib < B; ++ib) {
    for (int t = 0; t < L; ++t) {
      const float* x_bt = x + (ib * L + t) * D;
      for (int h = 0; h < H; ++h) {
        float q_raw = b_qkv[h], k_raw = b_qkv[h + H], v_raw = b_qkv[h + 2 * H];
        for (int d = 0; d < D; ++d) {
          const float xv = x_bt[d];
          q_raw += xv * W_qkv[h * D + d];
          k_raw += xv * W_qkv[(h + H) * D + d];
          v_raw += xv * W_qkv[(h + 2 * H) * D + d];
        }
        Q[(ib * L + t) * H + h] = q_raw;
        K[(ib * L + t) * H + h] = k_raw;
        V[(ib * L + t) * H + h] = v_raw;
      }
    }
  }

  for (int ib = 0; ib < B; ++ib) {
    for (int h = 0; h < H; ++h) {
      float z_hist[64] = {0.0f};
      int ksize = (k_size <= 64) ? k_size : 64;
      for (int t = 0; t < L; ++t) {
        float zp = b_z[h];
        const float* x_bt = x + (ib * L + t) * D;
        for (int d = 0; d < D; ++d) zp += x_bt[d] * W_z[h * D + d];
        z_hist[t % ksize] = zp;
        float z_conv = 0.0f;
        int valid = (t + 1 < ksize) ? (t + 1) : ksize;
        for (int lag = 0; lag < valid; ++lag) {
          z_conv += z_hist[(t - lag) % ksize] * K_conv[lag * H + h];
        }
        float sigmoid = 1.0f / (1.0f + std::exp(-z_conv));
        V[(ib * L + t) * H + h] *= (z_conv * sigmoid);
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

  std::vector<std::tuple<int, int, int, int, int>> test_cases = {
      {1, 128, 64, 32, 4},
      {1, 256, 128, 64, 4},
      {2, 512, 256, 128, 4},
      {4, 1024, 512, 256, 4},
      {8, 2048, 512, 256, 4},
  };

  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/fused_conv1d_silu_v2_results.csv");
  ofs << "B,L,D,H,k_size,cpu_ms,gpu_ms_v2,max_abs_diff_q,max_abs_diff_k,"
         "max_abs_diff_v,check\n";

  std::cout << "=== Fused Conv1D + SiLU V2 (Dual Kernel + vec4) ===\n";
  std::cout << std::left << std::setw(6) << "B" << std::setw(6) << "L"
            << std::setw(6) << "D" << std::setw(6) << "H" << std::setw(8) << "k_size"
            << std::setw(14) << "CPU ms" << std::setw(14) << "GPU ms"
            << std::setw(10) << "SpdUp" << std::setw(9) << "Check" << "\n";
  std::cout << std::string(73, '-') << "\n";

  for (const auto& tc : test_cases) {
    int B = std::get<0>(tc);
    int L = std::get<1>(tc);
    int D = std::get<2>(tc);
    int H = std::get<3>(tc);
    int k_size = std::get<4>(tc);

    std::vector<float> h_x(B * L * D);
    std::vector<float> h_W_qkv(3 * H * D), h_b_qkv(3 * H);
    std::vector<float> h_W_z(H * D), h_b_z(H);
    std::vector<float> h_K_conv(k_size * H);
    std::vector<float> h_Q_cpu(B * L * H), h_K_cpu(B * L * H), h_V_cpu(B * L * H);
    std::vector<float> h_Q_gpu(B * L * H), h_K_gpu(B * L * H), h_V_gpu(B * L * H);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto rand_fill = [&](std::vector<float>& v) {
      for (auto& x : v) x = dist(gen);
    };
    rand_fill(h_x);
    rand_fill(h_W_qkv); rand_fill(h_b_qkv);
    rand_fill(h_W_z); rand_fill(h_b_z);
    rand_fill(h_K_conv);

    auto t0 = std::chrono::high_resolution_clock::now();
    FusedConv1dSiLU_CPU(h_x.data(),
                        h_W_qkv.data(), h_b_qkv.data(),
                        h_W_z.data(), h_b_z.data(),
                        h_K_conv.data(),
                        h_Q_cpu.data(), h_K_cpu.data(), h_V_cpu.data(),
                        B, L, D, H, k_size);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    float *d_x, *d_W_qkv, *d_b_qkv, *d_W_z, *d_b_z, *d_K_conv;
    float *d_Q, *d_K, *d_V, *d_z_proj;

    CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W_qkv, h_W_qkv.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b_qkv, h_b_qkv.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W_z, h_W_z.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b_z, h_b_z.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K_conv, h_K_conv.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Q, h_Q_gpu.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, h_K_gpu.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, h_V_gpu.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_z_proj, h_Q_gpu.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_qkv, h_W_qkv.data(), h_W_qkv.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_qkv, h_b_qkv.data(), h_b_qkv.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_z, h_W_z.data(), h_W_z.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_z, h_b_z.data(), h_b_z.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K_conv, h_K_conv.data(), h_K_conv.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int total_blks = (B * L * H + fused_v2::kBlockSize - 1) / fused_v2::kBlockSize;

    for (int w = 0; w < kWarmup; ++w) {
      fused_v2::ComputeQKVZKernel<<<total_blks, fused_v2::kBlockSize>>>(
          d_x, d_W_qkv, d_b_qkv, d_W_z, d_b_z,
          d_Q, d_K, d_V, d_z_proj, B, L, D, H);
      fused_v2::ConvGateKernel<<<total_blks, fused_v2::kBlockSize>>>(
          d_z_proj, d_K_conv, d_V, B, L, H, k_size);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));

    std::vector<float> gpu_times;
    for (int rep = 0; rep < kRepeat; ++rep) {
      CHECK_CUDA(cudaEventRecord(s));
      fused_v2::ComputeQKVZKernel<<<total_blks, fused_v2::kBlockSize>>>(
          d_x, d_W_qkv, d_b_qkv, d_W_z, d_b_z,
          d_Q, d_K, d_V, d_z_proj, B, L, D, H);
      fused_v2::ConvGateKernel<<<total_blks, fused_v2::kBlockSize>>>(
          d_z_proj, d_K_conv, d_V, B, L, H, k_size);
      CHECK_CUDA(cudaEventRecord(e));
      CHECK_CUDA(cudaEventSynchronize(e));
      CHECK_CUDA(cudaGetLastError());
      float ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
      gpu_times.push_back(ms);
    }

    std::sort(gpu_times.begin(), gpu_times.end());
    float gpu_ms = 0.0f;
    if (gpu_times.size() > 2) {
      for (size_t t = 1; t + 1 < gpu_times.size(); ++t) gpu_ms += gpu_times[t];
      gpu_ms /= static_cast<float>(gpu_times.size() - 2);
    } else {
      for (float t : gpu_times) gpu_ms += t;
      gpu_ms /= static_cast<float>(gpu_times.size());
    }

    CHECK_CUDA(cudaMemcpy(h_Q_gpu.data(), d_Q, h_Q_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_K_gpu.data(), d_K, h_K_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_V_gpu.data(), d_V, h_V_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double max_diff_q = common::MaxAbsDiff(h_Q_cpu, h_Q_gpu);
    double max_diff_k = common::MaxAbsDiff(h_K_cpu, h_K_gpu);
    double max_diff_v = common::MaxAbsDiff(h_V_cpu, h_V_gpu);
    bool ok = (max_diff_q < 1e-3 && max_diff_k < 1e-3 && max_diff_v < 1e-3);
    const char* check = ok ? "PASS" : "FAIL";

    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_W_qkv)); CHECK_CUDA(cudaFree(d_b_qkv));
    CHECK_CUDA(cudaFree(d_W_z)); CHECK_CUDA(cudaFree(d_b_z));
    CHECK_CUDA(cudaFree(d_K_conv));
    CHECK_CUDA(cudaFree(d_Q)); CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V)); CHECK_CUDA(cudaFree(d_z_proj));

    double speedup = (gpu_ms > 0) ? cpu_ms / gpu_ms : 0.0;
    std::cout << std::left << std::setw(6) << B << std::setw(6) << L
              << std::setw(6) << D << std::setw(6) << H << std::setw(8) << k_size
              << std::fixed << std::setprecision(3) << std::setw(14) << cpu_ms
              << std::setw(14) << gpu_ms
              << std::setw(10) << std::setprecision(2) << speedup
              << std::setw(9) << check << "\n";

    ofs << B << "," << L << "," << D << "," << H << "," << k_size << ","
        << cpu_ms << "," << gpu_ms << ","
        << max_diff_q << "," << max_diff_k << "," << max_diff_v << ","
        << check << "\n";
  }

  std::cout << "\nResults saved to data/results/fused_conv1d_silu_v2_results.csv\n";
  return 0;
}
