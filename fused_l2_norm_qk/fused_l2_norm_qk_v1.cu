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

// ============================================================================
// Fused L2 Norm Q/K v1: Warp shuffle reduction + float4 vectorization
// ============================================================================
// Key improvements over v0:
//   1. Warp shuffle reduction replaces SMEM tree reduction:
//      - No __syncthreads() needed inside reduction (just __syncwarp())
//      - Only 1 __syncthreads() for the warp-level partial sums
//      - Eliminates SMEM bandwidth bottleneck for reduction
//   2. Float4 vectorized loads/stores where H is divisible by 4
//   3. Removed s_sum[256] shared memory — lower SMEM usage, better occupancy
// ============================================================================

constexpr float kEps = 1e-6f;
constexpr int kWarpSize = 32;

// ----------------------------------------------------------------------------
// Kernel 1: L2 Normalize using warp shuffle + float4
// grid(B, N), blockDim(128) — each block processes one row
// ----------------------------------------------------------------------------
__global__ void L2NormShuffleKernel(const float* __restrict__ X,
                                    float* __restrict__ X_hat,
                                    int B, int N, int H) {
  int b = blockIdx.x;
  int n = blockIdx.y;
  if (b >= B || n >= N) return;

  int row_offset = (b * N + n) * H;
  const float* x_row = X + row_offset;
  float* out_row = X_hat + row_offset;

  int tid = threadIdx.x;
  int warp_id = tid / kWarpSize;
  int lane = tid % kWarpSize;
  int num_warps = blockDim.x / kWarpSize;

  // Step 1: Sum of squares — each thread accumulates its elements
  float local_sum = 0.0f;

  // Float4 vectorized accumulation
  int H4 = H / 4;
  for (int i = tid; i < H4; i += blockDim.x) {
    float4 v = *reinterpret_cast<const float4*>(x_row + i * 4);
    local_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  // Handle tail elements
  for (int i = H4 * 4 + tid; i < H; i += blockDim.x) {
    float val = x_row[i];
    local_sum += val * val;
  }

  // Step 2: Warp-level shuffle reduction
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
  }

  // Step 3: Warp-level partial sums to shared memory, then reduce across warps
  __shared__ float s_warp_sum[4];  // max 4 warps with 128 threads
  if (lane == 0) {
    s_warp_sum[warp_id] = local_sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    float warp_sum = (lane < num_warps) ? s_warp_sum[lane] : 0.0f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      warp_sum += __shfl_xor_sync(0xFFFFFFFF, warp_sum, offset);
    }
    if (lane == 0) {
      s_warp_sum[0] = warp_sum;
    }
  }
  __syncthreads();

  float norm = sqrtf(s_warp_sum[0] + kEps);
  float rcp_norm = 1.0f / norm;

  // Step 4: Write normalized values with float4 vectorization
  for (int i = tid; i < H4; i += blockDim.x) {
    float4 v = *reinterpret_cast<const float4*>(x_row + i * 4);
    v.x *= rcp_norm; v.y *= rcp_norm; v.z *= rcp_norm; v.w *= rcp_norm;
    *reinterpret_cast<float4*>(out_row + i * 4) = v;
  }
  for (int i = H4 * 4 + tid; i < H; i += blockDim.x) {
    out_row[i] = x_row[i] * rcp_norm;
  }
}

// ============================================================================
// CPU Reference Implementation
// ============================================================================
static void L2Norm_CPU(const float* X, float* X_hat, int B, int N, int H) {
  for (int b = 0; b < B; ++b) {
    for (int n = 0; n < N; ++n) {
      int offset = (b * N + n) * H;
      float sum_sq = 0.0f;
      for (int h = 0; h < H; ++h) {
        float val = X[offset + h];
        sum_sq += val * val;
      }
      float norm = std::sqrt(sum_sq + kEps);
      for (int h = 0; h < H; ++h) {
        X_hat[offset + h] = X[offset + h] / norm;
      }
    }
  }
}

// ============================================================================
// GPU v1 Implementation
// ============================================================================
static void RunGpuV1(const float* d_Q, const float* d_K,
                     float* d_Q_hat, float* d_K_hat,
                     int B, int N_q, int H_q, int N_k, int H_k) {
  dim3 block(128);
  dim3 grid_q(B, N_q);
  dim3 grid_k(B, N_k);

  L2NormShuffleKernel<<<grid_q, block>>>(d_Q, d_Q_hat, B, N_q, H_q);
  L2NormShuffleKernel<<<grid_k, block>>>(d_K, d_K_hat, B, N_k, H_k);
}

// ============================================================================
// Main
// ============================================================================
int main() {
  constexpr int kWarmup = 1;
  constexpr int kRepeat = 10;

  std::vector<std::tuple<int, int, int, int, int>> test_cases = {
      {1, 128, 64, 128, 64},
      {1, 256, 128, 256, 128},
      {2, 512, 128, 512, 128},
      {4, 1024, 256, 1024, 256},
      {8, 2048, 256, 2048, 256},
  };

  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/fused_l2_norm_qk_v1_results.csv");
  ofs << "B,N_q,H_q,N_k,H_k,cpu_ms,gpu_ms,speedup,max_abs_diff_q,max_abs_diff_k,check\n";

  std::cout << "=== Fused L2 Norm Q/K V1 (Warp Shuffle + float4) ===\n";
  std::cout << std::left << std::setw(6) << "B"
            << std::setw(8) << "N_q" << std::setw(8) << "H_q"
            << std::setw(8) << "N_k" << std::setw(8) << "H_k"
            << std::setw(14) << "GPU ms" << std::setw(10) << "Speedup"
            << std::setw(8) << "Check" << "\n";
  std::cout << std::string(64, '-') << "\n";

  for (const auto& tc : test_cases) {
    int B = std::get<0>(tc);
    int N_q = std::get<1>(tc);
    int H_q = std::get<2>(tc);
    int N_k = std::get<3>(tc);
    int H_k = std::get<4>(tc);

    std::vector<float> h_Q(B * N_q * H_q);
    std::vector<float> h_K(B * N_k * H_k);
    std::vector<float> h_Q_hat_cpu(B * N_q * H_q);
    std::vector<float> h_K_hat_cpu(B * N_k * H_k);
    std::vector<float> h_Q_hat_gpu(B * N_q * H_q);
    std::vector<float> h_K_hat_gpu(B * N_k * H_k);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto rand_fill = [&](std::vector<float>& v) {
      for (auto& x : v) x = dist(gen);
    };
    rand_fill(h_Q);
    rand_fill(h_K);

    // CPU reference
    auto t0 = std::chrono::high_resolution_clock::now();
    L2Norm_CPU(h_Q.data(), h_Q_hat_cpu.data(), B, N_q, H_q);
    L2Norm_CPU(h_K.data(), h_K_hat_cpu.data(), B, N_k, H_k);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    float *d_Q, *d_K, *d_Q_hat, *d_K_hat;
    CHECK_CUDA(cudaMalloc(&d_Q, h_Q.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, h_K.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Q_hat, h_Q_hat_gpu.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K_hat, h_K_hat_gpu.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), h_Q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), h_K.size() * sizeof(float), cudaMemcpyHostToDevice));

    for (int w = 0; w < kWarmup; ++w) {
      RunGpuV1(d_Q, d_K, d_Q_hat, d_K_hat, B, N_q, H_q, N_k, H_k);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    std::vector<float> gpu_times;
    gpu_times.reserve(kRepeat);

    for (int rep = 0; rep < kRepeat; ++rep) {
      CHECK_CUDA(cudaEventRecord(s));
      RunGpuV1(d_Q, d_K, d_Q_hat, d_K_hat, B, N_q, H_q, N_k, H_k);
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

    CHECK_CUDA(cudaMemcpy(h_Q_hat_gpu.data(), d_Q_hat, h_Q_hat_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_K_hat_gpu.data(), d_K_hat, h_K_hat_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double max_diff_q = common::MaxAbsDiff(h_Q_hat_cpu, h_Q_hat_gpu);
    double max_diff_k = common::MaxAbsDiff(h_K_hat_cpu, h_K_hat_gpu);
    bool ok = (max_diff_q < 1e-4f && max_diff_k < 1e-4f);
    const char* check = ok ? "PASS" : "FAIL";

    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_Q_hat));
    CHECK_CUDA(cudaFree(d_K_hat));

    double speedup = (gpu_ms > 0) ? cpu_ms / gpu_ms : 0;
    std::cout << std::left << std::setw(6) << B
              << std::setw(8) << N_q << std::setw(8) << H_q
              << std::setw(8) << N_k << std::setw(8) << H_k
              << std::fixed << std::setprecision(4) << std::setw(14) << gpu_ms
              << std::setw(10) << std::setprecision(2) << speedup
              << std::setw(8) << check << "\n";

    ofs << B << "," << N_q << "," << H_q << "," << N_k << "," << H_k << ","
        << cpu_ms << "," << gpu_ms << "," << speedup << ","
        << max_diff_q << "," << max_diff_k << "," << check << "\n";
  }

  std::cout << "\nResults saved to data/results/fused_l2_norm_qk_v1_results.csv\n";
  return 0;
}
