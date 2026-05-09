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
// Fused Gated Delta Rule v1: Fused projections (2 kernels)
// ============================================================================
// v1 merges the 3 separate LinearActivationKernels into 1 kernel.
// Each thread still computes one (b,t,h) triple, but does all 3 projections
// (decay, delta, state) in a single pass, reusing the same x_bt row.
//
// Benefits vs v0:
//   1. 3 kernel launches -> 1 (saves CUDA launch overhead)
//   2. x_bt loaded once per thread, reused across 3 dot products
//   3. Better L1/L2 cache utilization (same data accessed 3× in quick succession)
// ============================================================================

constexpr float kEps = 1e-6f;

// ----------------------------------------------------------------------------
// Kernel 1: Fused Linear Projections + Activations
// Each thread computes all 3 projections for one (b,t,h) position
// ----------------------------------------------------------------------------
__global__ void FusedProjectionKernel(const float* __restrict__ x,
                                      const float* __restrict__ W_decay,
                                      const float* __restrict__ b_decay,
                                      const float* __restrict__ W_delta,
                                      const float* __restrict__ b_delta,
                                      const float* __restrict__ W_state,
                                      const float* __restrict__ b_state,
                                      float* __restrict__ alpha,
                                      float* __restrict__ delta,
                                      float* __restrict__ u,
                                      int B, int L, int D, int H) {
  int b_idx = blockIdx.x;
  int t = blockIdx.y;
  int base_h = blockIdx.z * blockDim.x + threadIdx.x;
  if (b_idx >= B || t >= L) return;

  const float* x_bt = x + (b_idx * L + t) * D;

  for (int h = base_h; h < H; h += gridDim.z * blockDim.x) {
    float a = b_decay[h];
    float d = b_delta[h];
    float v = b_state[h];

    const float* Wd = W_decay + h * D;
    const float* Wdl = W_delta + h * D;
    const float* Ws = W_state + h * D;

    for (int i = 0; i < D; ++i) {
      float xi = x_bt[i];
      a += xi * Wd[i];
      d += xi * Wdl[i];
      v += xi * Ws[i];
    }

    alpha[(b_idx * L + t) * H + h] = 1.0f / (1.0f + expf(-a));

    float abs_d = fabsf(d);
    delta[(b_idx * L + t) * H + h] = fmaxf(d, 0.0f) + logf(1.0f + expf(-abs_d));

    u[(b_idx * L + t) * H + h] = v;
  }
}

// ----------------------------------------------------------------------------
// Kernel 2: Recurrent Delta Rule Update (same as v0)
// ----------------------------------------------------------------------------
__global__ void RecurrentDeltaRuleKernel(const float* __restrict__ alpha,
                                         const float* __restrict__ delta,
                                         const float* __restrict__ u,
                                         float* __restrict__ output,
                                         int B, int L, int H) {
  int b_idx = blockIdx.x;
  int base_h = blockIdx.y * blockDim.x + threadIdx.x;
  if (b_idx >= B) return;

  for (int h = base_h; h < H; h += gridDim.y * blockDim.x) {
    float s = 0.0f;
    for (int t = 0; t < L; ++t) {
      int idx = (b_idx * L + t) * H + h;
      s = alpha[idx] * s + delta[idx] * u[idx];
      output[idx] = s;
    }
  }
}

// ============================================================================
// CPU Reference Implementation (same as v0)
// ============================================================================
static void GatedDeltaRule_CPU(const float* x,
                               const float* W_decay, const float* b_decay,
                               const float* W_delta, const float* b_delta,
                               const float* W_state, const float* b_state,
                               float* output,
                               int B, int L, int D, int H) {
  std::vector<float> alpha(B * L * H), delta(B * L * H), u(B * L * H);

  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < L; ++t) {
      const float* x_bt = x + (b * L + t) * D;
      for (int h = 0; h < H; ++h) {
        float a = b_decay[h];
        for (int d = 0; d < D; ++d) a += x_bt[d] * W_decay[h * D + d];
        alpha[(b * L + t) * H + h] = 1.0f / (1.0f + std::exp(-a));

        float dlt = b_delta[h];
        for (int d = 0; d < D; ++d) dlt += x_bt[d] * W_delta[h * D + d];
        dlt = std::max(dlt, 0.0f) + std::log(1.0f + std::exp(-std::abs(dlt)));
        delta[(b * L + t) * H + h] = dlt;

        float v = b_state[h];
        for (int d = 0; d < D; ++d) v += x_bt[d] * W_state[h * D + d];
        u[(b * L + t) * H + h] = v;
      }
    }
  }

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
// GPU v1 Implementation (fused projections)
// ============================================================================
static void RunGpuV1(const float* d_x,
                     const float* d_W_decay, const float* d_b_decay,
                     const float* d_W_delta, const float* d_b_delta,
                     const float* d_W_state, const float* d_b_state,
                     float* d_alpha, float* d_delta, float* d_u,
                     float* d_output,
                     int B, int L, int D, int H) {
  int threads = 256;
  int h_per_tb = (H + threads - 1) / threads;
  dim3 grid(B, L, h_per_tb);
  FusedProjectionKernel<<<grid, threads>>>(d_x,
      d_W_decay, d_b_decay, d_W_delta, d_b_delta, d_W_state, d_b_state,
      d_alpha, d_delta, d_u, B, L, D, H);

  int h_blocks = (H + threads - 1) / threads;
  dim3 grid_rec(B, h_blocks);
  RecurrentDeltaRuleKernel<<<grid_rec, threads>>>(d_alpha, d_delta, d_u,
                                                   d_output, B, L, H);
}

// ============================================================================
// Main
// ============================================================================
int main() {
  constexpr int kWarmup = 1;
  constexpr int kRepeat = 10;

  std::vector<std::tuple<int, int, int, int>> test_cases = {
      {1, 128, 64, 32},
      {1, 256, 128, 64},
      {2, 512, 256, 128},
      {4, 1024, 512, 256},
      {8, 2048, 512, 256},
  };

  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/fused_gated_delta_rule_v1_results.csv");
  ofs << "B,L,D,H,cpu_ms,gpu_ms,speedup,max_abs_diff,check\n";

  std::cout << "=== Fused Gated Delta Rule V1 (Fused Projections) ===\n";
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

    std::vector<float> h_x(B * L * D);
    std::vector<float> h_W_decay(H * D), h_b_decay(H);
    std::vector<float> h_W_delta(H * D), h_b_delta(H);
    std::vector<float> h_W_state(H * D), h_b_state(H);
    std::vector<float> h_output_cpu(B * L * H);
    std::vector<float> h_output_gpu(B * L * H);

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

    // Device memory
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

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_decay, h_W_decay.data(), h_W_decay.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_decay, h_b_decay.data(), h_b_decay.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_delta, h_W_delta.data(), h_W_delta.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_delta, h_b_delta.data(), h_b_delta.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_state, h_W_state.data(), h_W_state.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_state, h_b_state.data(), h_b_state.size() * sizeof(float), cudaMemcpyHostToDevice));

    for (int w = 0; w < kWarmup; ++w) {
      RunGpuV1(d_x, d_W_decay, d_b_decay, d_W_delta, d_b_delta,
               d_W_state, d_b_state, d_alpha, d_delta, d_u, d_output,
               B, L, D, H);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    std::vector<float> gpu_times;
    gpu_times.reserve(kRepeat);

    for (int rep = 0; rep < kRepeat; ++rep) {
      CHECK_CUDA(cudaEventRecord(s));
      RunGpuV1(d_x, d_W_decay, d_b_decay, d_W_delta, d_b_delta,
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

    CHECK_CUDA(cudaMemcpy(h_output_gpu.data(), d_output, h_output_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double max_diff = common::MaxAbsDiff(h_output_cpu, h_output_gpu);
    bool ok = (max_diff < 1e-3f);
    const char* check = ok ? "PASS" : "FAIL";

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

  std::cout << "\nResults saved to data/results/fused_gated_delta_rule_v1_results.csv\n";
  return 0;
}
