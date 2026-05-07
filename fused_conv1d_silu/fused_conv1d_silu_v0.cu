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
// Fused Conv1D + SiLU v0: Naive baseline (separate kernels per op)
// ============================================================================
// Operations:
//   1. Linear(in_proj_qkv): x(B,L,D) @ W_qkv(D,3H) + b_qkv -> qkv(B,L,3H)
//   2. Linear(in_proj_z):   x(B,L,D) @ W_z(D,H)   + b_z   -> z(B,L,H)
//   3. Linear(in_proj_a):   x(B,L,D) @ W_a(D,H)   + b_a   -> a(B,L,H)
//   4. Linear(in_proj_b):   x(B,L,D) @ W_b(D,H)   + b_b   -> b(B,L,H)
//   5. Split qkv -> Q_raw, K_raw, V_raw (each B,L,H)
//   6. CausalConv1d(z) + SiLU -> z_act
//   7. CausalConv1d(a) + SiLU -> a_act
//   8. CausalConv1d(b) + SiLU -> b_act
//   9. V = V_raw * z_act
// ============================================================================

// ----------------------------------------------------------------------------
// Kernel 1: Linear projection (GEMV per token)
// ----------------------------------------------------------------------------
__global__ void LinearKernel(const float* __restrict__ x,
                             const float* __restrict__ W,
                             const float* __restrict__ b,
                             float* __restrict__ out,
                             int B, int L, int D, int H_out) {
  int b_idx = blockIdx.x;
  int t = blockIdx.y;
  int h = threadIdx.x + blockIdx.z * blockDim.x;
  if (b_idx >= B || t >= L || h >= H_out) return;

  float sum = (b != nullptr) ? b[h] : 0.0f;
  const float* x_bt = x + (b_idx * L + t) * D;
  const float* W_h = W + h * D;
  for (int d = 0; d < D; ++d) {
    sum += x_bt[d] * W_h[d];
  }
  out[(b_idx * L + t) * H_out + h] = sum;
}

// ----------------------------------------------------------------------------
// Kernel 2: Split qkv into Q, K, V
// ----------------------------------------------------------------------------
__global__ void SplitQKVKernel(const float* __restrict__ qkv,
                                float* __restrict__ Q,
                                float* __restrict__ K,
                                float* __restrict__ V,
                                int B, int L, int H) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * L * H;
  if (idx >= total) return;

  int b = idx / (L * H);
  int rem = idx % (L * H);
  int t = rem / H;
  int h = rem % H;

  int base = ((b * L) + t) * (3 * H);
  Q[idx] = qkv[base + h];
  K[idx] = qkv[base + H + h];
  V[idx] = qkv[base + 2 * H + h];
}

// ----------------------------------------------------------------------------
// Kernel 3: Causal Conv1D + SiLU
// Input: u(B,L,H), kernel K(k_size, H)
// Output: y(B,L,H) where y[t] = SiLU(sum_{i=0}^{min(t,k-1)} K[i] * u[t-i])
// ----------------------------------------------------------------------------
__global__ void CausalConv1dSiLUKernel(const float* __restrict__ u,
                                       const float* __restrict__ K_conv,
                                       float* __restrict__ y,
                                       int B, int L, int H, int k_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * L * H;
  if (idx >= total) return;

  int b = idx / (L * H);
  int rem = idx % (L * H);
  int t = rem / H;
  int h = rem % H;

  float sum = 0.0f;
  int k_start = (t < k_size) ? 0 : (t - k_size + 1);
  for (int i = k_start; i <= t; ++i) {
    int u_idx = ((b * L) + i) * H + h;
    int k_idx = (t - i) * H + h;
    sum += u[u_idx] * K_conv[k_idx];
  }

  // SiLU(x) = x * sigmoid(x)
  float sigmoid = 1.0f / (1.0f + expf(-sum));
  y[idx] = sum * sigmoid;
}

// ----------------------------------------------------------------------------
// Kernel 4: Element-wise multiply V = V_raw * z_act
// ----------------------------------------------------------------------------
__global__ void GateMulKernel(const float* __restrict__ V_raw,
                              const float* __restrict__ gate,
                              float* __restrict__ V_out,
                              int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  V_out[idx] = V_raw[idx] * gate[idx];
}

// ----------------------------------------------------------------------------
// Kernel 5: Single fused kernel (qkv + z-conv-silu + v gate)
// Mapping: one thread handles one (b, h), loops over t.
// ----------------------------------------------------------------------------
__global__ void FusedSingleKernel(const float* __restrict__ x,
                                  const float* __restrict__ W_qkv,
                                  const float* __restrict__ b_qkv,
                                  const float* __restrict__ W_z,
                                  const float* __restrict__ b_z,
                                  const float* __restrict__ K_conv,
                                  float* __restrict__ Q,
                                  float* __restrict__ K,
                                  float* __restrict__ V,
                                  int B, int L, int D, int H, int k_size) {
  constexpr int kMaxKernelSize = 64;
  const int b_idx = blockIdx.x;
  const int h = blockIdx.y * blockDim.x + threadIdx.x;
  if (b_idx >= B || h >= H || k_size <= 0 || k_size > kMaxKernelSize) return;

  float z_hist[kMaxKernelSize];
  for (int i = 0; i < k_size; ++i) z_hist[i] = 0.0f;

  const float* Wq = W_qkv + static_cast<size_t>(h) * static_cast<size_t>(D);
  const float* Wk = W_qkv + static_cast<size_t>(h + H) * static_cast<size_t>(D);
  const float* Wv = W_qkv + static_cast<size_t>(h + 2 * H) * static_cast<size_t>(D);
  const float* Wz = W_z + static_cast<size_t>(h) * static_cast<size_t>(D);

  for (int t = 0; t < L; ++t) {
    const float* x_bt = x + (static_cast<size_t>(b_idx) * static_cast<size_t>(L) + t) * D;

    float q_raw = b_qkv[h];
    float k_raw = b_qkv[h + H];
    float v_raw = b_qkv[h + 2 * H];
    float z_proj = b_z[h];
    for (int d = 0; d < D; ++d) {
      const float xv = x_bt[d];
      q_raw += xv * Wq[d];
      k_raw += xv * Wk[d];
      v_raw += xv * Wv[d];
      z_proj += xv * Wz[d];
    }

    z_hist[t % k_size] = z_proj;
    float z_conv = 0.0f;
    const int valid = (t + 1 < k_size) ? (t + 1) : k_size;
    for (int lag = 0; lag < valid; ++lag) {
      const int hist_idx = (t - lag) % k_size;
      z_conv += z_hist[hist_idx] * K_conv[static_cast<size_t>(lag) * static_cast<size_t>(H) + h];
    }
    const float sigmoid = 1.0f / (1.0f + expf(-z_conv));
    const float z_act = z_conv * sigmoid;

    const size_t out_idx =
        (static_cast<size_t>(b_idx) * static_cast<size_t>(L) + t) * static_cast<size_t>(H) + h;
    Q[out_idx] = q_raw;
    K[out_idx] = k_raw;
    V[out_idx] = v_raw * z_act;
  }
}

// ============================================================================
// CPU Reference Implementation
// ============================================================================
static void FusedConv1dSiLU_CPU(const float* x,
                                const float* W_qkv, const float* b_qkv,
                                const float* W_z, const float* b_z,
                                const float* W_a, const float* b_a,
                                const float* W_b, const float* b_b,
                                const float* K_conv,
                                float* Q, float* K, float* V,
                                int B, int L, int D, int H, int k_size) {
  std::vector<float> qkv(B * L * 3 * H);
  std::vector<float> z(B * L * H), a(B * L * H), b_gate(B * L * H);

  // Linear projections
  for (int ib = 0; ib < B; ++ib) {
    for (int t = 0; t < L; ++t) {
      const float* x_bt = x + (ib * L + t) * D;

      // qkv projection
      for (int h = 0; h < 3 * H; ++h) {
        float sum = b_qkv[h];
        for (int d = 0; d < D; ++d) sum += x_bt[d] * W_qkv[h * D + d];
        qkv[(ib * L + t) * 3 * H + h] = sum;
      }
      // z, a, b projections
      for (int h = 0; h < H; ++h) {
        float sz = b_z[h], sa = b_a[h], sb = b_b[h];
        for (int d = 0; d < D; ++d) {
          sz += x_bt[d] * W_z[h * D + d];
          sa += x_bt[d] * W_a[h * D + d];
          sb += x_bt[d] * W_b[h * D + d];
        }
        z[(ib * L + t) * H + h] = sz;
        a[(ib * L + t) * H + h] = sa;
        b_gate[(ib * L + t) * H + h] = sb;
      }
    }
  }

  // Split QKV
  for (int ib = 0; ib < B; ++ib) {
    for (int t = 0; t < L; ++t) {
      for (int h = 0; h < H; ++h) {
        int base = (ib * L + t) * 3 * H;
        Q[(ib * L + t) * H + h] = qkv[base + h];
        K[(ib * L + t) * H + h] = qkv[base + H + h];
        V[(ib * L + t) * H + h] = qkv[base + 2 * H + h];
      }
    }
  }

  // Causal Conv1D + SiLU for z, a, b
  auto causal_conv_silu = [&](const std::vector<float>& u, std::vector<float>& out) {
    for (int ib = 0; ib < B; ++ib) {
      for (int t = 0; t < L; ++t) {
        for (int h = 0; h < H; ++h) {
          float sum = 0.0f;
          int k_start = (t < k_size) ? 0 : (t - k_size + 1);
          for (int i = k_start; i <= t; ++i) {
            sum += u[(ib * L + i) * H + h] * K_conv[(t - i) * H + h];
          }
          float sigmoid = 1.0f / (1.0f + std::exp(-sum));
          out[(ib * L + t) * H + h] = sum * sigmoid;
        }
      }
    }
  };

  std::vector<float> z_act(B * L * H), a_act(B * L * H), b_act(B * L * H);
  causal_conv_silu(z, z_act);
  causal_conv_silu(a, a_act);
  causal_conv_silu(b_gate, b_act);

  // V = V_raw * z_act
  for (int i = 0; i < B * L * H; ++i) {
    V[i] = V[i] * z_act[i];
  }
}

// ============================================================================
// GPU Naive Implementation (separate kernels)
// ============================================================================
static void RunGpuV0(const float* d_x,
                     const float* d_W_qkv, const float* d_b_qkv,
                     const float* d_W_z, const float* d_b_z,
                     const float* d_W_a, const float* d_b_a,
                     const float* d_W_b, const float* d_b_b,
                     const float* d_K_conv,
                     float* d_Q, float* d_K, float* d_V,
                     float* d_qkv, float* d_z, float* d_a, float* d_b,
                     float* d_z_act, float* d_a_act, float* d_b_act,
                     int B, int L, int D, int H, int k_size) {
  // 1. Linear projections
  dim3 block_lin(256);
  dim3 grid_qkv(B, L, (3 * H + 255) / 256);
  LinearKernel<<<grid_qkv, block_lin>>>(d_x, d_W_qkv, d_b_qkv, d_qkv, B, L, D, 3 * H);

  dim3 grid_gate(B, L, (H + 255) / 256);
  LinearKernel<<<grid_gate, block_lin>>>(d_x, d_W_z, d_b_z, d_z, B, L, D, H);
  LinearKernel<<<grid_gate, block_lin>>>(d_x, d_W_a, d_b_a, d_a, B, L, D, H);
  LinearKernel<<<grid_gate, block_lin>>>(d_x, d_W_b, d_b_b, d_b, B, L, D, H);

  // 2. Split QKV
  int total_elements = B * L * H;
  int block_split = 256;
  int grid_split = (total_elements + block_split - 1) / block_split;
  SplitQKVKernel<<<grid_split, block_split>>>(d_qkv, d_Q, d_K, d_V, B, L, H);

  // 3. Causal Conv1D + SiLU
  int total_gate = B * L * H;
  int block_conv = 256;
  int grid_conv = (total_gate + block_conv - 1) / block_conv;
  CausalConv1dSiLUKernel<<<grid_conv, block_conv>>>(d_z, d_K_conv, d_z_act, B, L, H, k_size);
  CausalConv1dSiLUKernel<<<grid_conv, block_conv>>>(d_a, d_K_conv, d_a_act, B, L, H, k_size);
  CausalConv1dSiLUKernel<<<grid_conv, block_conv>>>(d_b, d_K_conv, d_b_act, B, L, H, k_size);

  // 4. Gate multiply V = V_raw * z_act
  int total_v = B * L * H;
  int block_gate = 256;
  int grid_gatemul = (total_v + block_gate - 1) / block_gate;
  GateMulKernel<<<grid_gatemul, block_gate>>>(d_V, d_z_act, d_V, total_v);
}

static void RunGpuV1SingleKernel(const float* d_x,
                                 const float* d_W_qkv, const float* d_b_qkv,
                                 const float* d_W_z, const float* d_b_z,
                                 const float* d_K_conv,
                                 float* d_Q, float* d_K, float* d_V,
                                 int B, int L, int D, int H, int k_size) {
  const int block_size = 256;
  dim3 block(block_size);
  dim3 grid(B, (H + block_size - 1) / block_size);
  FusedSingleKernel<<<grid, block>>>(
      d_x, d_W_qkv, d_b_qkv, d_W_z, d_b_z, d_K_conv, d_Q, d_K, d_V, B, L, D, H, k_size);
}

// ============================================================================
// Main
// ============================================================================
int main() {
  constexpr int kWarmup = 1;
  constexpr int kRepeat = 10;

  // Test configurations: (B, L, D, H, k_size)
  std::vector<std::tuple<int, int, int, int, int>> test_cases = {
      {1, 128, 64, 32, 4},
      {1, 256, 128, 64, 4},
      {2, 512, 256, 128, 4},
      {4, 1024, 512, 256, 4},
      {8, 2048, 512, 256, 4},
  };

  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/fused_conv1d_silu_v0_results.csv");
  ofs << "B,L,D,H,k_size,cpu_ms,gpu_ms_v0,gpu_ms_v1,speedup_v0,speedup_v1,"
         "max_abs_diff_q_v0,max_abs_diff_k_v0,max_abs_diff_v_v0,check_v0,"
         "max_abs_diff_q_v1,max_abs_diff_k_v1,max_abs_diff_v_v1,check_v1\n";

  std::cout << "=== Fused Conv1D + SiLU V0 (Naive Separate Kernels) ===\n";
  std::cout << std::left << std::setw(6) << "B" << std::setw(6) << "L"
            << std::setw(6) << "D" << std::setw(6) << "H" << std::setw(8) << "k_size"
            << std::setw(14) << "V0 ms" << std::setw(14) << "V1 ms"
            << std::setw(10) << "V0x" << std::setw(10) << "V1x"
            << std::setw(9) << "V0Chk" << std::setw(9) << "V1Chk" << "\n";
  std::cout << std::string(86, '-') << "\n";

  for (const auto& tc : test_cases) {
    int B = std::get<0>(tc);
    int L = std::get<1>(tc);
    int D = std::get<2>(tc);
    int H = std::get<3>(tc);
    int k_size = std::get<4>(tc);

    // Allocate host memory
    std::vector<float> h_x(B * L * D);
    std::vector<float> h_W_qkv(3 * H * D), h_b_qkv(3 * H);
    std::vector<float> h_W_z(H * D), h_b_z(H);
    std::vector<float> h_W_a(H * D), h_b_a(H);
    std::vector<float> h_W_b(H * D), h_b_b(H);
    std::vector<float> h_K_conv(k_size * H);

    std::vector<float> h_Q_cpu(B * L * H), h_K_cpu(B * L * H), h_V_cpu(B * L * H);
    std::vector<float> h_Q_gpu_v0(B * L * H), h_K_gpu_v0(B * L * H), h_V_gpu_v0(B * L * H);
    std::vector<float> h_Q_gpu_v1(B * L * H), h_K_gpu_v1(B * L * H), h_V_gpu_v1(B * L * H);

    // Initialize with random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto rand_fill = [&](std::vector<float>& v) {
      for (auto& x : v) x = dist(gen);
    };
    rand_fill(h_x);
    rand_fill(h_W_qkv); rand_fill(h_b_qkv);
    rand_fill(h_W_z); rand_fill(h_b_z);
    rand_fill(h_W_a); rand_fill(h_b_a);
    rand_fill(h_W_b); rand_fill(h_b_b);
    rand_fill(h_K_conv);

    // CPU reference
    auto t0 = std::chrono::high_resolution_clock::now();
    FusedConv1dSiLU_CPU(h_x.data(),
                        h_W_qkv.data(), h_b_qkv.data(),
                        h_W_z.data(), h_b_z.data(),
                        h_W_a.data(), h_b_a.data(),
                        h_W_b.data(), h_b_b.data(),
                        h_K_conv.data(),
                        h_Q_cpu.data(), h_K_cpu.data(), h_V_cpu.data(),
                        B, L, D, H, k_size);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Allocate device memory
    float *d_x, *d_W_qkv, *d_b_qkv, *d_W_z, *d_b_z, *d_W_a, *d_b_a, *d_W_b, *d_b_b;
    float *d_K_conv;
    float *d_Q, *d_K, *d_V;
    float *d_qkv, *d_z, *d_a, *d_b;
    float *d_z_act, *d_a_act, *d_b_act;

    CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W_qkv, h_W_qkv.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b_qkv, h_b_qkv.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W_z, h_W_z.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b_z, h_b_z.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W_a, h_W_a.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b_a, h_b_a.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W_b, h_W_b.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b_b, h_b_b.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K_conv, h_K_conv.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Q, h_Q_gpu_v0.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, h_K_gpu_v0.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, h_V_gpu_v0.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_qkv, B * L * 3 * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_z, B * L * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_a, B * L * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, B * L * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_z_act, B * L * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_a_act, B * L * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b_act, B * L * H * sizeof(float)));

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_qkv, h_W_qkv.data(), h_W_qkv.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_qkv, h_b_qkv.data(), h_b_qkv.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_z, h_W_z.data(), h_W_z.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_z, h_b_z.data(), h_b_z.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_a, h_W_a.data(), h_W_a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_a, h_b_a.data(), h_b_a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_b, h_W_b.data(), h_W_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_b, h_b_b.data(), h_b_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K_conv, h_K_conv.data(), h_K_conv.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Warmup
    for (int w = 0; w < kWarmup; ++w) {
      RunGpuV0(d_x, d_W_qkv, d_b_qkv, d_W_z, d_b_z, d_W_a, d_b_a, d_W_b, d_b_b, d_K_conv,
               d_Q, d_K, d_V, d_qkv, d_z, d_a, d_b, d_z_act, d_a_act, d_b_act,
               B, L, D, H, k_size);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark v0
    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    std::vector<float> gpu_times_v0;
    gpu_times_v0.reserve(kRepeat);

    for (int rep = 0; rep < kRepeat; ++rep) {
      CHECK_CUDA(cudaEventRecord(s));
      RunGpuV0(d_x, d_W_qkv, d_b_qkv, d_W_z, d_b_z, d_W_a, d_b_a, d_W_b, d_b_b, d_K_conv,
               d_Q, d_K, d_V, d_qkv, d_z, d_a, d_b, d_z_act, d_a_act, d_b_act,
               B, L, D, H, k_size);
      CHECK_CUDA(cudaEventRecord(e));
      CHECK_CUDA(cudaEventSynchronize(e));
      CHECK_CUDA(cudaGetLastError());
      float ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
      gpu_times_v0.push_back(ms);
    }

    std::sort(gpu_times_v0.begin(), gpu_times_v0.end());
    float gpu_ms_v0 = 0.0f;
    if (gpu_times_v0.size() > 2) {
      for (size_t t = 1; t + 1 < gpu_times_v0.size(); ++t) gpu_ms_v0 += gpu_times_v0[t];
      gpu_ms_v0 /= static_cast<float>(gpu_times_v0.size() - 2);
    } else {
      for (float t : gpu_times_v0) gpu_ms_v0 += t;
      gpu_ms_v0 /= static_cast<float>(gpu_times_v0.size());
    }

    CHECK_CUDA(cudaMemcpy(h_Q_gpu_v0.data(), d_Q, h_Q_gpu_v0.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_K_gpu_v0.data(), d_K, h_K_gpu_v0.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_V_gpu_v0.data(), d_V, h_V_gpu_v0.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Warmup v1 single-kernel
    for (int w = 0; w < kWarmup; ++w) {
      RunGpuV1SingleKernel(
          d_x, d_W_qkv, d_b_qkv, d_W_z, d_b_z, d_K_conv, d_Q, d_K, d_V, B, L, D, H, k_size);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark v1 single-kernel
    std::vector<float> gpu_times_v1;
    gpu_times_v1.reserve(kRepeat);
    for (int rep = 0; rep < kRepeat; ++rep) {
      CHECK_CUDA(cudaEventRecord(s));
      RunGpuV1SingleKernel(
          d_x, d_W_qkv, d_b_qkv, d_W_z, d_b_z, d_K_conv, d_Q, d_K, d_V, B, L, D, H, k_size);
      CHECK_CUDA(cudaEventRecord(e));
      CHECK_CUDA(cudaEventSynchronize(e));
      CHECK_CUDA(cudaGetLastError());
      float ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
      gpu_times_v1.push_back(ms);
    }

    std::sort(gpu_times_v1.begin(), gpu_times_v1.end());
    float gpu_ms_v1 = 0.0f;
    if (gpu_times_v1.size() > 2) {
      for (size_t t = 1; t + 1 < gpu_times_v1.size(); ++t) gpu_ms_v1 += gpu_times_v1[t];
      gpu_ms_v1 /= static_cast<float>(gpu_times_v1.size() - 2);
    } else {
      for (float t : gpu_times_v1) gpu_ms_v1 += t;
      gpu_ms_v1 /= static_cast<float>(gpu_times_v1.size());
    }
    CHECK_CUDA(cudaMemcpy(h_Q_gpu_v1.data(), d_Q, h_Q_gpu_v1.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_K_gpu_v1.data(), d_K, h_K_gpu_v1.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_V_gpu_v1.data(), d_V, h_V_gpu_v1.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double max_diff_q_v0 = common::MaxAbsDiff(h_Q_cpu, h_Q_gpu_v0);
    double max_diff_k_v0 = common::MaxAbsDiff(h_K_cpu, h_K_gpu_v0);
    double max_diff_v_v0 = common::MaxAbsDiff(h_V_cpu, h_V_gpu_v0);
    bool ok_v0 = (max_diff_q_v0 < 1e-3f && max_diff_k_v0 < 1e-3f && max_diff_v_v0 < 1e-3f);
    const char* check_v0 = ok_v0 ? "PASS" : "FAIL";

    double max_diff_q_v1 = common::MaxAbsDiff(h_Q_cpu, h_Q_gpu_v1);
    double max_diff_k_v1 = common::MaxAbsDiff(h_K_cpu, h_K_gpu_v1);
    double max_diff_v_v1 = common::MaxAbsDiff(h_V_cpu, h_V_gpu_v1);
    bool ok_v1 = (max_diff_q_v1 < 1e-3f && max_diff_k_v1 < 1e-3f && max_diff_v_v1 < 1e-3f);
    const char* check_v1 = ok_v1 ? "PASS" : "FAIL";

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_W_qkv)); CHECK_CUDA(cudaFree(d_b_qkv));
    CHECK_CUDA(cudaFree(d_W_z)); CHECK_CUDA(cudaFree(d_b_z));
    CHECK_CUDA(cudaFree(d_W_a)); CHECK_CUDA(cudaFree(d_b_a));
    CHECK_CUDA(cudaFree(d_W_b)); CHECK_CUDA(cudaFree(d_b_b));
    CHECK_CUDA(cudaFree(d_K_conv));
    CHECK_CUDA(cudaFree(d_Q)); CHECK_CUDA(cudaFree(d_K)); CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_qkv)); CHECK_CUDA(cudaFree(d_z)); CHECK_CUDA(cudaFree(d_a)); CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_z_act)); CHECK_CUDA(cudaFree(d_a_act)); CHECK_CUDA(cudaFree(d_b_act));

    double speedup_v0 = (gpu_ms_v0 > 0) ? cpu_ms / gpu_ms_v0 : 0;
    double speedup_v1 = (gpu_ms_v1 > 0) ? cpu_ms / gpu_ms_v1 : 0;
    std::cout << std::left << std::setw(6) << B << std::setw(6) << L
              << std::setw(6) << D << std::setw(6) << H << std::setw(8) << k_size
              << std::fixed << std::setprecision(4) << std::setw(14) << gpu_ms_v0
              << std::setw(14) << gpu_ms_v1
              << std::setw(10) << std::setprecision(2) << speedup_v0
              << std::setw(10) << std::setprecision(2) << speedup_v1
              << std::setw(9) << check_v0
              << std::setw(9) << check_v1 << "\n";

    ofs << B << "," << L << "," << D << "," << H << "," << k_size << ","
        << cpu_ms << "," << gpu_ms_v0 << "," << gpu_ms_v1 << ","
        << speedup_v0 << "," << speedup_v1 << ","
        << max_diff_q_v0 << "," << max_diff_k_v0 << "," << max_diff_v_v0 << "," << check_v0
        << "," << max_diff_q_v1 << "," << max_diff_k_v1 << "," << max_diff_v_v1 << ","
        << check_v1 << "\n";
  }

  std::cout << "\nResults saved to data/results/fused_conv1d_silu_v0_results.csv\n";
  return 0;
}
