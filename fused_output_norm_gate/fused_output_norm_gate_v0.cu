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
// Fused Output Norm Gate v0: Naive baseline (separate kernels per step)
// ============================================================================
// Operations:
//   1. Gate_Projection + SiLU: x(B,L,D_in) -> gate(B,L,H)
//   2. RMSNorm: gate(B,L,H) -> x_hat(B,L,H)
//   3. Multiply_Gate: x_hat * gate -> y(B,L,H)
//   4. Linear(out_proj): y(B,L,H) -> output(B,L,D_out)
// ============================================================================

constexpr float kEps = 1e-6f;

// ----------------------------------------------------------------------------
// Kernel 1: Gate Projection + SiLU
// Each block processes one (b, t) token, threads compute output features
// ----------------------------------------------------------------------------
__global__ void GateProjectionSiLUKernel(const float* __restrict__ x,
                                         const float* __restrict__ W_gate,
                                         const float* __restrict__ b_gate,
                                         float* __restrict__ gate,
                                         int B, int L, int D_in, int H) {
  int b_idx = blockIdx.x;
  int t = blockIdx.y;
  int h = threadIdx.x + blockIdx.z * blockDim.x;
  if (b_idx >= B || t >= L || h >= H) return;

  float sum = b_gate[h];
  const float* x_bt = x + (b_idx * L + t) * D_in;
  const float* W_h = W_gate + h * D_in;
  for (int d = 0; d < D_in; ++d) {
    sum += x_bt[d] * W_h[d];
  }

  // SiLU(x) = x * sigmoid(x)
  float sigmoid = 1.0f / (1.0f + expf(-sum));
  gate[(b_idx * L + t) * H + h] = sum * sigmoid;
}

// ----------------------------------------------------------------------------
// Kernel 2: RMSNorm
// Each block processes one (b, t) row, threads cooperate to compute RMS
// ----------------------------------------------------------------------------
__global__ void RMSNormKernel(const float* __restrict__ gate,
                              const float* __restrict__ g,
                              float* __restrict__ x_hat,
                              int B, int L, int H) {
  int b_idx = blockIdx.x;
  int t = blockIdx.y;
  if (b_idx >= B || t >= L) return;

  int row_offset = (b_idx * L + t) * H;
  const float* gate_row = gate + row_offset;
  float* out_row = x_hat + row_offset;

  __shared__ float s_sum[256];
  float local_sum = 0.0f;
  int tid = threadIdx.x;

  for (int h = tid; h < H; h += blockDim.x) {
    float val = gate_row[h];
    local_sum += val * val;
  }
  s_sum[tid] = local_sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (tid < offset) {
      s_sum[tid] += s_sum[tid + offset];
    }
    __syncthreads();
  }

  float rms = sqrtf(s_sum[0] / H + kEps);

  for (int h = tid; h < H; h += blockDim.x) {
    out_row[h] = (gate_row[h] / rms) * g[h];
  }
}

// ----------------------------------------------------------------------------
// Kernel 3: Multiply Gate (element-wise)
// ----------------------------------------------------------------------------
__global__ void MultiplyGateKernel(const float* __restrict__ x_hat,
                                   const float* __restrict__ gate,
                                   float* __restrict__ y,
                                   int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  y[idx] = x_hat[idx] * gate[idx];
}

// ----------------------------------------------------------------------------
// Kernel 4: Linear Output Projection
// Each block processes one (b, t) token, threads compute output features
// ----------------------------------------------------------------------------
__global__ void LinearOutputKernel(const float* __restrict__ y,
                                   const float* __restrict__ W_out,
                                   const float* __restrict__ b_out,
                                   float* __restrict__ output,
                                   int B, int L, int H, int D_out) {
  int b_idx = blockIdx.x;
  int t = blockIdx.y;
  int d = threadIdx.x + blockIdx.z * blockDim.x;
  if (b_idx >= B || t >= L || d >= D_out) return;

  float sum = b_out[d];
  const float* y_bt = y + (b_idx * L + t) * H;
  const float* W_d = W_out + d * H;
  for (int h = 0; h < H; ++h) {
    sum += y_bt[h] * W_d[h];
  }
  output[(b_idx * L + t) * D_out + d] = sum;
}

// ============================================================================
// CPU Reference Implementation
// ============================================================================
static void OutputNormGate_CPU(const float* x,
                               const float* W_gate, const float* b_gate,
                               const float* g,
                               const float* W_out, const float* b_out,
                               float* output,
                               int B, int L, int D_in, int H, int D_out) {
  std::vector<float> gate(B * L * H);
  std::vector<float> x_hat(B * L * H);
  std::vector<float> y(B * L * H);

  // Step 1: Gate Projection + SiLU
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < L; ++t) {
      const float* x_bt = x + (b * L + t) * D_in;
      for (int h = 0; h < H; ++h) {
        float sum = b_gate[h];
        for (int d = 0; d < D_in; ++d) {
          sum += x_bt[d] * W_gate[h * D_in + d];
        }
        // SiLU
        float sigmoid = 1.0f / (1.0f + std::exp(-sum));
        gate[(b * L + t) * H + h] = sum * sigmoid;
      }
    }
  }

  // Step 2: RMSNorm
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < L; ++t) {
      int offset = (b * L + t) * H;
      float sum_sq = 0.0f;
      for (int h = 0; h < H; ++h) {
        sum_sq += gate[offset + h] * gate[offset + h];
      }
      float rms = std::sqrt(sum_sq / H + kEps);
      for (int h = 0; h < H; ++h) {
        x_hat[offset + h] = (gate[offset + h] / rms) * g[h];
      }
    }
  }

  // Step 3: Multiply Gate
  for (int i = 0; i < B * L * H; ++i) {
    y[i] = x_hat[i] * gate[i];
  }

  // Step 4: Linear Output Projection
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < L; ++t) {
      const float* y_bt = y.data() + (b * L + t) * H;
      for (int d = 0; d < D_out; ++d) {
        float sum = b_out[d];
        for (int h = 0; h < H; ++h) {
          sum += y_bt[h] * W_out[d * H + h];
        }
        output[(b * L + t) * D_out + d] = sum;
      }
    }
  }
}

// ============================================================================
// GPU Naive Implementation (separate kernels)
// ============================================================================
static void RunGpuV0(const float* d_x,
                     const float* d_W_gate, const float* d_b_gate,
                     const float* d_g,
                     const float* d_W_out, const float* d_b_out,
                     float* d_gate, float* d_x_hat, float* d_y,
                     float* d_output,
                     int B, int L, int D_in, int H, int D_out) {
  dim3 block(256);

  // 1. Gate Projection + SiLU
  dim3 grid_gate(B, L, (H + 255) / 256);
  GateProjectionSiLUKernel<<<grid_gate, block>>>(d_x, d_W_gate, d_b_gate, d_gate,
                                                  B, L, D_in, H);

  // 2. RMSNorm
  dim3 grid_norm(B, L);
  RMSNormKernel<<<grid_norm, block>>>(d_gate, d_g, d_x_hat, B, L, H);

  // 3. Multiply Gate
  int total = B * L * H;
  int grid_mul = (total + 255) / 256;
  MultiplyGateKernel<<<grid_mul, block>>>(d_x_hat, d_gate, d_y, total);

  // 4. Linear Output Projection
  dim3 grid_out(B, L, (D_out + 255) / 256);
  LinearOutputKernel<<<grid_out, block>>>(d_y, d_W_out, d_b_out, d_output,
                                          B, L, H, D_out);
}

// ============================================================================
// Main
// ============================================================================
int main() {
  constexpr int kWarmup = 1;
  constexpr int kRepeat = 10;

  // Test configurations: (B, L, D_in, H, D_out)
  std::vector<std::tuple<int, int, int, int, int>> test_cases = {
      {1, 128, 64, 32, 64},
      {1, 256, 128, 64, 128},
      {2, 512, 256, 128, 256},
      {4, 1024, 512, 256, 512},
      {8, 2048, 512, 256, 512},
  };

  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/fused_output_norm_gate_v0_results.csv");
  ofs << "B,L,D_in,H,D_out,cpu_ms,gpu_ms,speedup,max_abs_diff,check\n";

  std::cout << "=== Fused Output Norm Gate V0 (Naive Separate Kernels) ===\n";
  std::cout << std::left << std::setw(6) << "B"
            << std::setw(8) << "L" << std::setw(8) << "D_in"
            << std::setw(8) << "H" << std::setw(8) << "D_out"
            << std::setw(14) << "GPU ms" << std::setw(10) << "Speedup"
            << std::setw(8) << "Check" << "\n";
  std::cout << std::string(72, '-') << "\n";

  for (const auto& tc : test_cases) {
    int B = std::get<0>(tc);
    int L = std::get<1>(tc);
    int D_in = std::get<2>(tc);
    int H = std::get<3>(tc);
    int D_out = std::get<4>(tc);

    // Allocate host memory
    std::vector<float> h_x(B * L * D_in);
    std::vector<float> h_W_gate(H * D_in), h_b_gate(H);
    std::vector<float> h_g(H);
    std::vector<float> h_W_out(D_out * H), h_b_out(D_out);
    std::vector<float> h_output_cpu(B * L * D_out);
    std::vector<float> h_output_gpu(B * L * D_out);

    // Initialize with random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto rand_fill = [&](std::vector<float>& v) {
      for (auto& x : v) x = dist(gen);
    };
    rand_fill(h_x);
    rand_fill(h_W_gate); rand_fill(h_b_gate);
    rand_fill(h_g);
    rand_fill(h_W_out); rand_fill(h_b_out);

    // CPU reference
    auto t0 = std::chrono::high_resolution_clock::now();
    OutputNormGate_CPU(h_x.data(),
                       h_W_gate.data(), h_b_gate.data(),
                       h_g.data(),
                       h_W_out.data(), h_b_out.data(),
                       h_output_cpu.data(),
                       B, L, D_in, H, D_out);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Allocate device memory
    float *d_x, *d_W_gate, *d_b_gate, *d_g, *d_W_out, *d_b_out;
    float *d_gate, *d_x_hat, *d_y, *d_output;

    CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W_gate, h_W_gate.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b_gate, h_b_gate.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g, h_g.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W_out, h_W_out.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b_out, h_b_out.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gate, B * L * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x_hat, B * L * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, B * L * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, h_output_gpu.size() * sizeof(float)));

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_gate, h_W_gate.data(), h_W_gate.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_gate, h_b_gate.data(), h_b_gate.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_g, h_g.data(), h_g.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_out, h_W_out.data(), h_W_out.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_out, h_b_out.data(), h_b_out.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Warmup
    for (int w = 0; w < kWarmup; ++w) {
      RunGpuV0(d_x, d_W_gate, d_b_gate, d_g, d_W_out, d_b_out,
               d_gate, d_x_hat, d_y, d_output,
               B, L, D_in, H, D_out);
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
      RunGpuV0(d_x, d_W_gate, d_b_gate, d_g, d_W_out, d_b_out,
               d_gate, d_x_hat, d_y, d_output,
               B, L, D_in, H, D_out);
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
    CHECK_CUDA(cudaFree(d_W_gate)); CHECK_CUDA(cudaFree(d_b_gate));
    CHECK_CUDA(cudaFree(d_g));
    CHECK_CUDA(cudaFree(d_W_out)); CHECK_CUDA(cudaFree(d_b_out));
    CHECK_CUDA(cudaFree(d_gate)); CHECK_CUDA(cudaFree(d_x_hat));
    CHECK_CUDA(cudaFree(d_y)); CHECK_CUDA(cudaFree(d_output));

    double speedup = (gpu_ms > 0) ? cpu_ms / gpu_ms : 0;
    std::cout << std::left << std::setw(6) << B
              << std::setw(8) << L << std::setw(8) << D_in
              << std::setw(8) << H << std::setw(8) << D_out
              << std::fixed << std::setprecision(4) << std::setw(14) << gpu_ms
              << std::setw(10) << std::setprecision(2) << speedup
              << std::setw(8) << check << "\n";

    ofs << B << "," << L << "," << D_in << "," << H << "," << D_out << ","
        << cpu_ms << "," << gpu_ms << "," << speedup << ","
        << max_diff << "," << check << "\n";
  }

  std::cout << "\nResults saved to data/results/fused_output_norm_gate_v0_results.csv\n";
  return 0;
}
