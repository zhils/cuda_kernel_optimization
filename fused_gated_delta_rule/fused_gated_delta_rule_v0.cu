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
// Fused Gated Delta Rule v0: Naive baseline (separate kernels per step)
// ============================================================================
// Operations:
//   1. Compute_Decay_Gate:  x(B,L,D) -> alpha(B,L,H)
//   2. Compute_Delta_Gate:  x(B,L,D) -> delta(B,L,H)
//   3. State_Projection:    x(B,L,D) -> u(B,L,H)
//   4. Recurrent_Update:    alpha, delta, u -> output(B,L,H)
//
// Recurrent formula: s_t = alpha_t * s_{t-1} + delta_t * u_t
//                    output_t = s_t
// ============================================================================

constexpr float kEps = 1e-6f;

// ----------------------------------------------------------------------------
// Kernel 1: Linear projection + activation (for decay/delta/state)
// Each block processes one (b, t) token, threads compute output features
// ----------------------------------------------------------------------------
__global__ void LinearActivationKernel(const float* __restrict__ x,
                                       const float* __restrict__ W,
                                       const float* __restrict__ b,
                                       float* __restrict__ out,
                                       int B, int L, int D, int H,
                                       int activation_type) {
  // activation_type: 0=none, 1=sigmoid, 2=softplus
  int b_idx = blockIdx.x;
  int t = blockIdx.y;
  int h = threadIdx.x + blockIdx.z * blockDim.x;
  if (b_idx >= B || t >= L || h >= H) return;

  float sum = b[h];
  const float* x_bt = x + (b_idx * L + t) * D;
  const float* W_h = W + h * D;
  for (int d = 0; d < D; ++d) {
    sum += x_bt[d] * W_h[d];
  }

  if (activation_type == 1) {
    // Sigmoid
    sum = 1.0f / (1.0f + expf(-sum));
  } else if (activation_type == 2) {
    // Softplus: ln(1 + exp(x))
    sum = fmaxf(sum, 0.0f) + logf(1.0f + expf(-fabsf(sum)));
  }

  out[(b_idx * L + t) * H + h] = sum;
}

// ----------------------------------------------------------------------------
// Kernel 2: Recurrent Delta Rule Update
// Each block processes one batch, threads process different features
// Sequential over time steps (causal)
// ----------------------------------------------------------------------------
__global__ void RecurrentDeltaRuleKernel(const float* __restrict__ alpha,
                                         const float* __restrict__ delta,
                                         const float* __restrict__ u,
                                         float* __restrict__ output,
                                         int B, int L, int H) {
  int b_idx = blockIdx.x;
  int h = threadIdx.x + blockIdx.y * blockDim.x;
  if (b_idx >= B || h >= H) return;

  float s = 0.0f;  // initial state

  for (int t = 0; t < L; ++t) {
    int idx = (b_idx * L + t) * H + h;
    float a = alpha[idx];
    float d = delta[idx];
    float v = u[idx];

    // s_t = alpha_t * s_{t-1} + delta_t * u_t
    s = a * s + d * v;
    output[idx] = s;
  }
}

// ============================================================================
// CPU Reference Implementation
// ============================================================================
static void GatedDeltaRule_CPU(const float* x,
                               const float* W_decay, const float* b_decay,
                               const float* W_delta, const float* b_delta,
                               const float* W_state, const float* b_state,
                               float* output,
                               int B, int L, int D, int H) {
  std::vector<float> alpha(B * L * H), delta(B * L * H), u(B * L * H);

  // Step 1-3: Linear projections + activations
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < L; ++t) {
      const float* x_bt = x + (b * L + t) * D;
      for (int h = 0; h < H; ++h) {
        // Decay gate (sigmoid)
        float a = b_decay[h];
        for (int d = 0; d < D; ++d) a += x_bt[d] * W_decay[h * D + d];
        a = 1.0f / (1.0f + std::exp(-a));
        alpha[(b * L + t) * H + h] = a;

        // Delta gate (softplus)
        float dlt = b_delta[h];
        for (int d = 0; d < D; ++d) dlt += x_bt[d] * W_delta[h * D + d];
        dlt = std::max(dlt, 0.0f) + std::log(1.0f + std::exp(-std::abs(dlt)));
        delta[(b * L + t) * H + h] = dlt;

        // State projection (no activation)
        float v = b_state[h];
        for (int d = 0; d < D; ++d) v += x_bt[d] * W_state[h * D + d];
        u[(b * L + t) * H + h] = v;
      }
    }
  }

  // Step 4: Recurrent update
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      float s = 0.0f;
      for (int t = 0; t < L; ++t) {
        int idx = (b * L + t) * H + h;
        s = alpha[idx] * s + delta[idx] * u[idx];
        output[idx] = s;
      }
    }
  }
}

// ============================================================================
// GPU Naive Implementation (separate kernels)
// ============================================================================
static void RunGpuV0(const float* d_x,
                     const float* d_W_decay, const float* d_b_decay,
                     const float* d_W_delta, const float* d_b_delta,
                     const float* d_W_state, const float* d_b_state,
                     float* d_alpha, float* d_delta, float* d_u,
                     float* d_output,
                     int B, int L, int D, int H) {
  dim3 block(256);
  dim3 grid(B, L, (H + 255) / 256);

  // 1. Decay gate (sigmoid)
  LinearActivationKernel<<<grid, block>>>(d_x, d_W_decay, d_b_decay, d_alpha,
                                          B, L, D, H, 1);
  // 2. Delta gate (softplus)
  LinearActivationKernel<<<grid, block>>>(d_x, d_W_delta, d_b_delta, d_delta,
                                          B, L, D, H, 2);
  // 3. State projection (no activation)
  LinearActivationKernel<<<grid, block>>>(d_x, d_W_state, d_b_state, d_u,
                                          B, L, D, H, 0);

  // 4. Recurrent update
  dim3 block_rec(256);
  dim3 grid_rec(B, (H + 255) / 256);
  RecurrentDeltaRuleKernel<<<grid_rec, block_rec>>>(d_alpha, d_delta, d_u,
                                                    d_output, B, L, H);
}

// ============================================================================
// Main
// ============================================================================
int main() {
  constexpr int kWarmup = 1;
  constexpr int kRepeat = 10;

  // Test configurations: (B, L, D, H)
  std::vector<std::tuple<int, int, int, int>> test_cases = {
      {1, 128, 64, 32},
      {1, 256, 128, 64},
      {2, 512, 256, 128},
      {4, 1024, 512, 256},
      {8, 2048, 512, 256},
  };

  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/fused_gated_delta_rule_v0_results.csv");
  ofs << "B,L,D,H,cpu_ms,gpu_ms,speedup,max_abs_diff,check\n";

  std::cout << "=== Fused Gated Delta Rule V0 (Naive Separate Kernels) ===\n";
  std::cout << std::left << std::setw(6) << "B"
            << std::setw(8) << "L" << std::setw(8) << "D"
            << std::setw(8) << "H"
            << std::setw(14) << "GPU ms" << std::setw(10) << "Speedup"
            << std::setw(8) << "Check" << "\n";
  std::cout << std::string(56, '-') << "\n";

  for (const auto& tc : test_cases) {
    int B = std::get<0>(tc);
    int L = std::get<1>(tc);
    int D = std::get<2>(tc);
    int H = std::get<3>(tc);

    // Allocate host memory
    std::vector<float> h_x(B * L * D);
    std::vector<float> h_W_decay(H * D), h_b_decay(H);
    std::vector<float> h_W_delta(H * D), h_b_delta(H);
    std::vector<float> h_W_state(H * D), h_b_state(H);
    std::vector<float> h_output_cpu(B * L * H);
    std::vector<float> h_output_gpu(B * L * H);

    // Initialize with random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto rand_fill = [&](std::vector<float>& v) {
      for (auto& x : v) x = dist(gen);
    };
    rand_fill(h_x);
    rand_fill(h_W_decay); rand_fill(h_b_decay);
    rand_fill(h_W_delta); rand_fill(h_b_delta);
    rand_fill(h_W_state); rand_fill(h_b_state);

    // CPU reference
    auto t0 = std::chrono::high_resolution_clock::now();
    GatedDeltaRule_CPU(h_x.data(),
                       h_W_decay.data(), h_b_decay.data(),
                       h_W_delta.data(), h_b_delta.data(),
                       h_W_state.data(), h_b_state.data(),
                       h_output_cpu.data(),
                       B, L, D, H);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Allocate device memory
    float *d_x, *d_W_decay, *d_b_decay, *d_W_delta, *d_b_delta;
    float *d_W_state, *d_b_state;
    float *d_alpha, *d_delta, *d_u, *d_output;

    CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W_decay, h_W_decay.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b_decay, h_b_decay.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W_delta, h_W_delta.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b_delta, h_b_delta.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W_state, h_W_state.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b_state, h_b_state.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_alpha, B * L * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_delta, B * L * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_u, B * L * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, h_output_gpu.size() * sizeof(float)));

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_decay, h_W_decay.data(), h_W_decay.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_decay, h_b_decay.data(), h_b_decay.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_delta, h_W_delta.data(), h_W_delta.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_delta, h_b_delta.data(), h_b_delta.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_state, h_W_state.data(), h_W_state.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_state, h_b_state.data(), h_b_state.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Warmup
    for (int w = 0; w < kWarmup; ++w) {
      RunGpuV0(d_x, d_W_decay, d_b_decay, d_W_delta, d_b_delta,
               d_W_state, d_b_state, d_alpha, d_delta, d_u, d_output,
               B, L, D, H);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    std::vector<float> gpu_times;
    gpu_times.reserve(kRepeat);

    for (int rep = 0; rep < kRepeat; ++rep) {
      CHECK_CUDA(cudaEventRecord(s));
      RunGpuV0(d_x, d_W_decay, d_b_decay, d_W_delta, d_b_delta,
               d_W_state, d_b_state, d_alpha, d_delta, d_u, d_output,
               B, L, D, H);
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

    // Copy back and verify
    CHECK_CUDA(cudaMemcpy(h_output_gpu.data(), d_output, h_output_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double max_diff = common::MaxAbsDiff(h_output_cpu, h_output_gpu);
    bool ok = (max_diff < 1e-3f);
    const char* check = ok ? "PASS" : "FAIL";

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_W_decay)); CHECK_CUDA(cudaFree(d_b_decay));
    CHECK_CUDA(cudaFree(d_W_delta)); CHECK_CUDA(cudaFree(d_b_delta));
    CHECK_CUDA(cudaFree(d_W_state)); CHECK_CUDA(cudaFree(d_b_state));
    CHECK_CUDA(cudaFree(d_alpha)); CHECK_CUDA(cudaFree(d_delta));
    CHECK_CUDA(cudaFree(d_u)); CHECK_CUDA(cudaFree(d_output));

    double speedup = (gpu_ms > 0) ? cpu_ms / gpu_ms : 0;
    std::cout << std::left << std::setw(6) << B
              << std::setw(8) << L << std::setw(8) << D
              << std::setw(8) << H
              << std::fixed << std::setprecision(4) << std::setw(14) << gpu_ms
              << std::setw(10) << std::setprecision(2) << speedup
              << std::setw(8) << check << "\n";

    ofs << B << "," << L << "," << D << "," << H << ","
        << cpu_ms << "," << gpu_ms << "," << speedup << ","
        << max_diff << "," << check << "\n";
  }

  std::cout << "\nResults saved to data/results/fused_gated_delta_rule_v0_results.csv\n";
  return 0;
}
