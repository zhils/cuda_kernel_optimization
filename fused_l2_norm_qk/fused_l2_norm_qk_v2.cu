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
// Fused L2 Norm Q/K v2: SMEM-cached row + single fused kernel
// ============================================================================
// v2 addresses the remaining bottleneck in v1: x_row is still read from
// global memory twice (once for sum-of-squares, once for normalization).
//
// Key improvements:
//   1. SMEM cache for x_row: load once from global into SMEM, then compute
//      both reduction and normalization from SMEM — cuts global reads in half
//   2. Single fused kernel for Q and K: one launch instead of two, gives
//      the GPU more blocks to schedule when individual grids are small
//   3. All optimizations from v1 retained: warp shuffle + float4
// ============================================================================

constexpr float kEps = 1e-6f;
constexpr int kWarpSize = 32;

// ----------------------------------------------------------------------------
// Single fused kernel: handles both Q and K in one launch
// blockIdx.x = batch * 2 + (0 for Q, 1 for K), blockIdx.y = row
// ----------------------------------------------------------------------------
__global__ void L2NormFusedKernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    float* __restrict__ Q_hat, float* __restrict__ K_hat,
    int B, int N_q, int H_q, int N_k, int H_k) {

  int total_q_rows = B * N_q;
  int total_k_rows = B * N_k;
  int total_rows = total_q_rows + total_k_rows;
  int row_id = blockIdx.x * gridDim.y + blockIdx.y;
  if (row_id >= total_rows) return;

  bool is_q = (row_id < total_q_rows);
  int b, n, H;
  const float* X;
  float* X_hat;

  if (is_q) {
    int flat_idx = row_id;
    b = flat_idx / N_q;
    n = flat_idx % N_q;
    H = H_q;
    X = Q;
    X_hat = Q_hat;
  } else {
    int flat_idx = row_id - total_q_rows;
    b = flat_idx / N_k;
    n = flat_idx % N_k;
    H = H_k;
    X = K;
    X_hat = K_hat;
  }

  int row_offset = (b * (is_q ? N_q : N_k) + n) * H;
  const float* x_row = X + row_offset;
  float* out_row = X_hat + row_offset;

  extern __shared__ float s_row[];
  int tid = threadIdx.x;

  // Cooperative load x_row into SMEM (float4 vectorized)
  int H4 = H / 4;
  for (int i = tid; i < H4; i += blockDim.x) {
    float4 v = *reinterpret_cast<const float4*>(x_row + i * 4);
    *reinterpret_cast<float4*>(s_row + i * 4) = v;
  }
  for (int i = H4 * 4 + tid; i < H; i += blockDim.x) {
    s_row[i] = x_row[i];
  }
  __syncthreads();

  // Step 1: Sum of squares from SMEM
  float local_sum = 0.0f;
  for (int i = tid; i < H4; i += blockDim.x) {
    float4 v = *reinterpret_cast<float4*>(s_row + i * 4);
    local_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  for (int i = H4 * 4 + tid; i < H; i += blockDim.x) {
    float val = s_row[i];
    local_sum += val * val;
  }

  // Warp shuffle reduction
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
  }

  // Cross-warp reduction via SMEM
  __shared__ float s_warp_sum[4];
  int warp_id = tid / kWarpSize;
  int lane = tid % kWarpSize;
  int num_warps = blockDim.x / kWarpSize;

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

  float rcp_norm = 1.0f / sqrtf(s_warp_sum[0] + kEps);

  // Step 2: Normalize from SMEM and write to global (float4)
  for (int i = tid; i < H4; i += blockDim.x) {
    float4 v = *reinterpret_cast<float4*>(s_row + i * 4);
    v.x *= rcp_norm; v.y *= rcp_norm; v.z *= rcp_norm; v.w *= rcp_norm;
    *reinterpret_cast<float4*>(out_row + i * 4) = v;
  }
  for (int i = H4 * 4 + tid; i < H; i += blockDim.x) {
    out_row[i] = s_row[i] * rcp_norm;
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
// GPU v2 Implementation
// ============================================================================
static void RunGpuV2(const float* d_Q, const float* d_K,
                     float* d_Q_hat, float* d_K_hat,
                     int B, int N_q, int H_q, int N_k, int H_k) {
  int total_q_rows = B * N_q;
  int total_k_rows = B * N_k;
  int total_rows = total_q_rows + total_k_rows;

  // Use 2D grid: x=num_row_chunks, y=rows_per_chunk
  // This handles up to ~65K rows (65535 per dim)
  int rows_per_chunk = 65535;
  int num_chunks = (total_rows + rows_per_chunk - 1) / rows_per_chunk;
  if (total_rows <= rows_per_chunk) {
    num_chunks = 1;
    rows_per_chunk = total_rows;
  }

  int H_max = std::max(H_q, H_k);
  size_t smem_size = H_max * sizeof(float);

  dim3 block(128);
  dim3 grid(num_chunks, rows_per_chunk);

  L2NormFusedKernel<<<grid, block, smem_size>>>(
      d_Q, d_K, d_Q_hat, d_K_hat, B, N_q, H_q, N_k, H_k);
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
  std::ofstream ofs("data/results/fused_l2_norm_qk_v2_results.csv");
  ofs << "B,N_q,H_q,N_k,H_k,cpu_ms,gpu_ms,speedup,max_abs_diff_q,max_abs_diff_k,check\n";

  std::cout << "=== Fused L2 Norm Q/K V2 (SMEM-cached + Fused Kernel) ===\n";
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
      RunGpuV2(d_Q, d_K, d_Q_hat, d_K_hat, B, N_q, H_q, N_k, H_k);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    std::vector<float> gpu_times;
    gpu_times.reserve(kRepeat);

    for (int rep = 0; rep < kRepeat; ++rep) {
      CHECK_CUDA(cudaEventRecord(s));
      RunGpuV2(d_Q, d_K, d_Q_hat, d_K_hat, B, N_q, H_q, N_k, H_k);
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

  std::cout << "\nResults saved to data/results/fused_l2_norm_qk_v2_results.csv\n";
  return 0;
}
