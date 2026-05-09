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
// Fused Output Norm Gate v1: Fused Gate + RMSNorm + Multiply (2 kernels)
// ============================================================================
// v1 merges steps 1-3 into a single kernel:
//   Gate_Proj + SiLU → RMSNorm → Multiply_Gate = one kernel
//
// Benefits vs v0:
//   1. 4 kernel launches → 2
//   2. Eliminates gate and x_hat intermediate buffers (only y remains)
//   3. RMSNorm uses warp shuffle instead of SMEM tree reduction
//   4. float4 vectorized loads for W
// ============================================================================

constexpr float kEps = 1e-6f;
constexpr int kWarpSize = 32;

// ----------------------------------------------------------------------------
// Kernel 1: Fused Gate Projection + SiLU + RMSNorm + Multiply Gate
// Each block handles one (b,t), threads handle h values with stride
// Output: y(B,L,H)
// ----------------------------------------------------------------------------
__global__ void FusedGateNormKernel(
    const float* __restrict__ x,
    const float* __restrict__ W_gate, const float* __restrict__ b_gate,
    const float* __restrict__ g,
    float* __restrict__ y,
    int B, int L, int D_in, int H) {

  int b = blockIdx.x;
  int t = blockIdx.y;
  if (b >= B || t >= L) return;

  const float* x_bt = x + (static_cast<size_t>(b) * L + t) * D_in;

  // Step 1: Gate Projection + SiLU, stored in SMEM for RMSNorm reduction
  extern __shared__ float s_gate[];
  int tid = threadIdx.x;
  int warp_id = tid / kWarpSize;
  int lane = tid % kWarpSize;
  int num_warps = blockDim.x / kWarpSize;

  float local_sum = 0.0f;

  // Each thread handles h values with stride
  for (int h = tid; h < H; h += blockDim.x) {
    const float* W_h = W_gate + h * D_in;
    float a = b_gate[h];
    for (int d = 0; d < D_in; ++d) {
      a += x_bt[d] * W_h[d];
    }
    a = a * (1.0f / (1.0f + expf(-a)));
    s_gate[h] = a;
    local_sum += a * a;
  }
  __syncthreads();

  // Step 2: RMSNorm via warp shuffle reduction
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
  }

  __shared__ float s_warp_sum[4];
  if (lane == 0) s_warp_sum[warp_id] = local_sum;
  __syncthreads();

  if (warp_id == 0) {
    float sum = (lane < num_warps) ? s_warp_sum[lane] : 0.0f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }
    if (lane == 0) s_warp_sum[0] = sum;
  }
  __syncthreads();

  float rms = sqrtf(s_warp_sum[0] / static_cast<float>(H) + kEps);
  float rcp_rms = 1.0f / rms;

  // Step 2b: Normalize + Step 3: Multiply Gate → write y
  for (int h = tid; h < H; h += blockDim.x) {
    float gate_h = s_gate[h];
    float x_hat_h = (gate_h * rcp_rms) * g[h];
    y[(static_cast<size_t>(b) * L + t) * H + h] = x_hat_h * gate_h;
  }
}

// ----------------------------------------------------------------------------
// Kernel 2: Linear Output Projection (float4 vectorized)
// ----------------------------------------------------------------------------
__global__ void LinearOutputV1Kernel(const float* __restrict__ y,
                                     const float* __restrict__ W_out,
                                     const float* __restrict__ b_out,
                                     float* __restrict__ output,
                                     int B, int L, int H, int D_out) {
  int b = blockIdx.x;
  int t = blockIdx.y;
  int base_d = blockIdx.z * blockDim.x + threadIdx.x;
  if (b >= B || t >= L) return;

  const float* y_bt = y + (static_cast<size_t>(b) * L + t) * H;

  for (int d = base_d; d < D_out; d += gridDim.z * blockDim.x) {
    float sum = b_out[d];
    const float* W_d = W_out + d * H;

    int H4 = H / 4;
    for (int i = 0; i < H4; ++i) {
      float4 yv = *reinterpret_cast<const float4*>(y_bt + i * 4);
      float4 wv = *reinterpret_cast<const float4*>(W_d + i * 4);
      sum += yv.x * wv.x + yv.y * wv.y + yv.z * wv.z + yv.w * wv.w;
    }
    for (int i = H4 * 4; i < H; ++i) {
      sum += y_bt[i] * W_d[i];
    }
    output[(static_cast<size_t>(b) * L + t) * D_out + d] = sum;
  }
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
  std::vector<float> y_vec(B * L * H);

  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < L; ++t) {
      const float* x_bt = x + (b * L + t) * D_in;
      for (int h = 0; h < H; ++h) {
        float sum = b_gate[h];
        for (int d = 0; d < D_in; ++d) sum += x_bt[d] * W_gate[h * D_in + d];
        float sig = 1.0f / (1.0f + std::exp(-sum));
        gate[(b * L + t) * H + h] = sum * sig;
      }
    }
  }

  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < L; ++t) {
      int offset = (b * L + t) * H;
      float sum_sq = 0.0f;
      for (int h = 0; h < H; ++h) sum_sq += gate[offset + h] * gate[offset + h];
      float rms = std::sqrt(sum_sq / H + kEps);
      for (int h = 0; h < H; ++h)
        x_hat[offset + h] = (gate[offset + h] / rms) * g[h];
    }
  }

  for (int i = 0; i < B * L * H; ++i)
    y_vec[i] = x_hat[i] * gate[i];

  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < L; ++t) {
      const float* y_bt = y_vec.data() + (b * L + t) * H;
      for (int d = 0; d < D_out; ++d) {
        float sum = b_out[d];
        for (int h = 0; h < H; ++h) sum += y_bt[h] * W_out[d * H + h];
        output[(b * L + t) * D_out + d] = sum;
      }
    }
  }
}

// ============================================================================
// GPU v1 Implementation
// ============================================================================
static void RunGpuV1(const float* d_x,
                     const float* d_W_gate, const float* d_b_gate,
                     const float* d_g,
                     const float* d_W_out, const float* d_b_out,
                     float* d_y,
                     float* d_output,
                     int B, int L, int D_in, int H, int D_out) {
  int threads = 128;
  size_t smem_size = H * sizeof(float);

  dim3 grid(B, L);
  FusedGateNormKernel<<<grid, threads, smem_size>>>(
      d_x, d_W_gate, d_b_gate, d_g, d_y, B, L, D_in, H);

  dim3 grid_out(B, L, (D_out + threads - 1) / threads);
  LinearOutputV1Kernel<<<grid_out, threads>>>(
      d_y, d_W_out, d_b_out, d_output, B, L, H, D_out);
}

// ============================================================================
// Main
// ============================================================================
int main() {
  constexpr int kWarmup = 1;
  constexpr int kRepeat = 10;

  std::vector<std::tuple<int, int, int, int, int>> test_cases = {
      {1, 128, 64, 32, 64},
      {1, 256, 128, 64, 128},
      {2, 512, 256, 128, 256},
      {4, 1024, 512, 256, 512},
      {8, 2048, 512, 256, 512},
  };

  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/fused_output_norm_gate_v1_results.csv");
  ofs << "B,L,D_in,H,D_out,cpu_ms,gpu_ms,speedup,max_abs_diff,check\n";

  std::cout << "=== Fused Output Norm Gate V1 (Fused Gate+Norm+Multiply) ===\n";
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

    std::vector<float> h_x(B * L * D_in);
    std::vector<float> h_W_gate(H * D_in), h_b_gate(H);
    std::vector<float> h_g(H);
    std::vector<float> h_W_out(D_out * H), h_b_out(D_out);
    std::vector<float> h_output_cpu(B * L * D_out);
    std::vector<float> h_output_gpu(B * L * D_out);

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
    OutputNormGate_CPU(h_x.data(), h_W_gate.data(), h_b_gate.data(),
                       h_g.data(), h_W_out.data(), h_b_out.data(),
                       h_output_cpu.data(), B, L, D_in, H, D_out);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    float *d_x, *d_W_gate, *d_b_gate, *d_g, *d_W_out, *d_b_out;
    float *d_y, *d_output;

    CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W_gate, h_W_gate.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b_gate, h_b_gate.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g, h_g.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W_out, h_W_out.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b_out, h_b_out.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, B * L * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, h_output_gpu.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_gate, h_W_gate.data(), h_W_gate.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_gate, h_b_gate.data(), h_b_gate.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_g, h_g.data(), h_g.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_out, h_W_out.data(), h_W_out.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_out, h_b_out.data(), h_b_out.size() * sizeof(float), cudaMemcpyHostToDevice));

    for (int w = 0; w < kWarmup; ++w) {
      RunGpuV1(d_x, d_W_gate, d_b_gate, d_g, d_W_out, d_b_out,
               d_y, d_output, B, L, D_in, H, D_out);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    std::vector<float> gpu_times;
    gpu_times.reserve(kRepeat);

    for (int rep = 0; rep < kRepeat; ++rep) {
      CHECK_CUDA(cudaEventRecord(s));
      RunGpuV1(d_x, d_W_gate, d_b_gate, d_g, d_W_out, d_b_out,
               d_y, d_output, B, L, D_in, H, D_out);
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
    CHECK_CUDA(cudaFree(d_W_gate)); CHECK_CUDA(cudaFree(d_b_gate));
    CHECK_CUDA(cudaFree(d_g));
    CHECK_CUDA(cudaFree(d_W_out)); CHECK_CUDA(cudaFree(d_b_out));
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

  std::cout << "\nResults saved to data/results/fused_output_norm_gate_v1_results.csv\n";
  return 0;
}
