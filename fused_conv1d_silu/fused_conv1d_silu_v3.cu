// Fused Conv1D + SiLU V3: 用 CUTLASS GEMM 替代自定义 GEMV 做 qkv/z 线性投影，
// ConvGate（因果 1D 卷积 + SiLU 门控）沿用 v2 的实现。

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"
// inline FusedConv1dSiLU_CPU defined below

static void FusedConv1dSiLU_CPU(
    const float* x,
    const float* W_qkv, const float* b_qkv,
    const float* W_z, const float* b_z,
    const float* K_conv,
    float* Q, float* K, float* V,
    int B, int L, int D, int H, int k_size) {
  std::vector<float> z_proj(static_cast<size_t>(B) * L * H);
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < L; ++t) {
      const float* x_bt = x + (static_cast<size_t>(b) * L + t) * D;
      for (int h = 0; h < H; ++h) {
        float q_raw = b_qkv[h];
        float k_raw = b_qkv[h + H];
        float v_raw = b_qkv[h + 2 * H];
        float zp = b_z[h];
        for (int d = 0; d < D; ++d) {
          float xv = x_bt[d];
          q_raw += xv * W_qkv[static_cast<size_t>(h) * D + d];
          k_raw += xv * W_qkv[static_cast<size_t>(h + H) * D + d];
          v_raw += xv * W_qkv[static_cast<size_t>(h + 2 * H) * D + d];
          zp += xv * W_z[static_cast<size_t>(h) * D + d];
        }
        size_t out_idx = (static_cast<size_t>(b) * L + t) * H + h;
        Q[out_idx] = q_raw;
        K[out_idx] = k_raw;
        V[out_idx] = v_raw;
        z_proj[out_idx] = zp;
      }
    }
  }
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < L; ++t) {
      for (int h = 0; h < H; ++h) {
        float z_conv = 0.0f;
        for (int i = 0; i < k_size; ++i) {
          int ti = t - i;
          if (ti < 0) continue;
          z_conv += z_proj[(static_cast<size_t>(b) * L + ti) * H + h] * K_conv[static_cast<size_t>(i) * H + h];
        }
        float sigmoid = 1.0f / (1.0f + std::exp(-z_conv));
        size_t out_idx = (static_cast<size_t>(b) * L + t) * H + h;
        V[out_idx] *= (z_conv * sigmoid);
      }
    }
  }
}

#include "fused_cutlass_gemm.cuh"

namespace fused_v3 {

constexpr int kBlockSize = 256;

// 将 GEMM 输出的 (BL, 3H) 行主序 qkv 拆分为三个 (BL, H) 张量并加 bias
__global__ void SplitQKVAddBiasKernel(
  const float* __restrict__ qkv,
  const float* __restrict__ b_qkv,
  float* __restrict__ Q,
  float* __restrict__ K,
  float* __restrict__ V,
  int BL, int H
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = BL * H;
  if (idx >= total) return;

  const int m = idx / H;
  const int h = idx - m * H;
  const size_t base = static_cast<size_t>(m) * (3 * H);

  Q[idx] = qkv[base + h] + b_qkv[h];
  K[idx] = qkv[base + H + h] + b_qkv[H + h];
  V[idx] = qkv[base + 2 * H + h] + b_qkv[2 * H + h];
}

// z 投影后加行广播 bias
__global__ void AddRowBiasKernel(float* __restrict__ z,
                                 const float* __restrict__ b,
                                 int BL, int H) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BL * H) return;
  z[idx] += b[idx % H];
}

// 因果 1D 卷积 + SiLU 门控，就地更新 V
__global__ void ConvGateKernel(
  const float* __restrict__ z_proj,
  const float* __restrict__ K_conv,
  float* __restrict__ V,
  int B,
  int L,
  int H,
  int k_size
) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_threads = blockDim.x * gridDim.x;

  // grid-stride loop 遍历所有 (b,t,h)
  for (int idx = global_idx; idx < B * L * H; idx += total_threads) {
    const int b = idx / (L * H);
    const int rem = idx % (L * H);
    const int t = rem / H;
    const int h = rem % H;

    const size_t base = (static_cast<size_t>(b) * L) * H;

    // 因果卷积
    float z_conv = 0.0f;
    for (int i = 0; i < k_size; ++i) {
      const int ti = t - i;
      if (ti < 0) continue;
      z_conv += z_proj[base + static_cast<size_t>(ti) * H + h] *
                K_conv[static_cast<size_t>(i) * H + h];
    }

    // SiLU 激活并写回 V
    const float sigmoid = 1.0f / (1.0f + expf(-z_conv));
    V[base + static_cast<size_t>(t) * H + h] *= (z_conv * sigmoid);
  }
}

inline void LaunchSplitQKV(const float* qkv, const float* b_qkv,
                           float* Q, float* K, float* V, int BL, int H) {
  const int grid = (BL * H + kBlockSize - 1) / kBlockSize;
  SplitQKVAddBiasKernel<<<grid, kBlockSize>>>(qkv, b_qkv, Q, K, V, BL, H);
}

inline void LaunchAddRowBias(float* z, const float* b, int BL, int H) {
  const int grid = (BL * H + kBlockSize - 1) / kBlockSize;
  AddRowBiasKernel<<<grid, kBlockSize>>>(z, b, BL, H);
}

struct GpuState {
  float *d_x = nullptr;
  float *d_W_qkv = nullptr;
  float *d_b_qkv = nullptr;
  float *d_W_z = nullptr;
  float *d_b_z = nullptr;
  float *d_K_conv = nullptr;
  float *d_Q = nullptr;
  float *d_K = nullptr;
  float *d_V = nullptr;
  float *d_qkv = nullptr;
  float *d_z_proj = nullptr;
};

inline void FreeGpuState(GpuState& s) {
  auto free_if = [](float*& p) {
    if (p) {
      CHECK_CUDA(cudaFree(p));
      p = nullptr;
    }
  };
  free_if(s.d_x);
  free_if(s.d_W_qkv);
  free_if(s.d_b_qkv);
  free_if(s.d_W_z);
  free_if(s.d_b_z);
  free_if(s.d_K_conv);
  free_if(s.d_Q);
  free_if(s.d_K);
  free_if(s.d_V);
  free_if(s.d_qkv);
  free_if(s.d_z_proj);
}

// 端到端 GPU 路径：
//   CUTLASS GEMM 做 qkv/z 线性投影，然后 SplitQKV / AddBias，最后 ConvGate
inline void RunFusedV3(fused_cutlass::GemmWorkspace& ws, GpuState& s,
                       int B, int L, int D, int H, int k_size) {
  const int BL = B * L;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // QKV 线性投影：x @ W_qkv^T
  fused_cutlass::LaunchGemmRowMajorWT(ws, BL, 3 * H, D, s.d_x, D, s.d_W_qkv, D, s.d_qkv,
                                      3 * H, alpha, beta);
  LaunchSplitQKV(s.d_qkv, s.d_b_qkv, s.d_Q, s.d_K, s.d_V, BL, H);

  // z 线性投影：x @ W_z^T
  fused_cutlass::LaunchGemmRowMajorWT(ws, BL, H, D, s.d_x, D, s.d_W_z, D, s.d_z_proj, H,
                                      alpha, beta);
  LaunchAddRowBias(s.d_z_proj, s.d_b_z, BL, H);

  // 因果卷积 + SiLU 门控
  const int total = B * L * H;
  const int grid = (total + kBlockSize - 1) / kBlockSize;
  ConvGateKernel<<<grid, kBlockSize>>>(s.d_z_proj, s.d_K_conv, s.d_V, B, L, H, k_size);
}

}  // namespace fused_v3

int main() {
  // 基准测试参数
  constexpr int kWarmup = 1;
  constexpr int kRepeat = 10;

  // 测试配置
  const std::vector<std::tuple<int, int, int, int, int>> test_cases = {
      {1, 128, 64, 32, 4},
      {1, 256, 128, 64, 4},
      {2, 512, 256, 128, 4},
      {4, 1024, 512, 256, 4},
      {8, 2048, 512, 256, 4},
  };

  // CUTLASS GEMM 工作空间
  fused_cutlass::GemmWorkspace gemm_ws;

  // 创建输出 CSV 文件并写入表头
  const std::string results_dir = common::EnsureResultsDir();
  std::ofstream ofs(results_dir + "/fused_conv1d_silu_v3_results.csv");
  ofs << "B,L,D,H,k_size,cpu_ms,gpu_ms_v3,max_abs_diff_q,max_abs_diff_k,"
         "max_abs_diff_v,check\n";

  std::cout << "=== Fused Conv1D + SiLU V3 (CUTLASS GEMM Projection) ===\n";
  std::cout << std::left << std::setw(6) << "B" << std::setw(6) << "L"
            << std::setw(6) << "D" << std::setw(6) << "H" << std::setw(8) << "k_size"
            << std::setw(14) << "CPU ms" << std::setw(14) << "GPU ms"
            << std::setw(10) << "SpdUp" << std::setw(9) << "Check" << "\n";
  std::cout << std::string(73, '-') << "\n";

  for (const auto& tc : test_cases) {
    // 提取测试参数
    const int B = std::get<0>(tc);
    const int L = std::get<1>(tc);
    const int D = std::get<2>(tc);
    const int H = std::get<3>(tc);
    const int k_size = std::get<4>(tc);
    const int BL = B * L;

    // 分配主机内存
    std::vector<float> h_x(B * L * D);
    std::vector<float> h_W_qkv(3 * H * D), h_b_qkv(3 * H);
    std::vector<float> h_W_z(H * D), h_b_z(H);
    std::vector<float> h_K_conv(k_size * H);
    std::vector<float> h_Q_cpu(B * L * H), h_K_cpu(B * L * H), h_V_cpu(B * L * H);
    std::vector<float> h_Q_gpu(B * L * H), h_K_gpu(B * L * H), h_V_gpu(B * L * H);

    // 随机初始化输入数据
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto rand_fill = [&](std::vector<float>& v) {
      for (auto& val : v) val = dist(gen);
    };
    rand_fill(h_x);
    rand_fill(h_W_qkv);
    rand_fill(h_b_qkv);
    rand_fill(h_W_z);
    rand_fill(h_b_z);
    rand_fill(h_K_conv);

    // 运行 CPU 参考实现并计时
    const auto t0 = std::chrono::high_resolution_clock::now();
    FusedConv1dSiLU_CPU(h_x.data(),
                        h_W_qkv.data(), h_b_qkv.data(),
                        h_W_z.data(), h_b_z.data(),
                        h_K_conv.data(),
                        h_Q_cpu.data(), h_K_cpu.data(), h_V_cpu.data(),
                        B, L, D, H, k_size);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 初始化 GPU 状态并分配内存
    fused_v3::GpuState gs;
    CHECK_CUDA(cudaMalloc(&gs.d_x, h_x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gs.d_W_qkv, h_W_qkv.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gs.d_b_qkv, h_b_qkv.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gs.d_W_z, h_W_z.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gs.d_b_z, h_b_z.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gs.d_K_conv, h_K_conv.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gs.d_Q, h_Q_gpu.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gs.d_K, h_K_gpu.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gs.d_V, h_V_gpu.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gs.d_qkv, static_cast<size_t>(BL) * (3 * H) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gs.d_z_proj, h_Q_gpu.size() * sizeof(float)));

    // 拷贝数据到 GPU
    CHECK_CUDA(cudaMemcpy(gs.d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gs.d_W_qkv, h_W_qkv.data(), h_W_qkv.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gs.d_b_qkv, h_b_qkv.data(), h_b_qkv.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gs.d_W_z, h_W_z.data(), h_W_z.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gs.d_b_z, h_b_z.data(), h_b_z.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gs.d_K_conv, h_K_conv.data(), h_K_conv.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    // GPU 预热
    for (int w = 0; w < kWarmup; ++w) {
      fused_v3::RunFusedV3(gemm_ws, gs, B, L, D, H, k_size);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // GPU 基准测试
    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));

    std::vector<float> gpu_times;
    for (int rep = 0; rep < kRepeat; ++rep) {
      CHECK_CUDA(cudaEventRecord(s));
      fused_v3::RunFusedV3(gemm_ws, gs, B, L, D, H, k_size);
      CHECK_CUDA(cudaEventRecord(e));
      CHECK_CUDA(cudaEventSynchronize(e));
      CHECK_CUDA(cudaGetLastError());
      float ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
      gpu_times.push_back(ms);
    }

    // 计算中位数耗时
    std::sort(gpu_times.begin(), gpu_times.end());
    float gpu_ms = 0.0f;
    if (gpu_times.size() > 2) {
      for (size_t i = 1; i + 1 < gpu_times.size(); ++i) gpu_ms += gpu_times[i];
      gpu_ms /= static_cast<float>(gpu_times.size() - 2);
    } else {
      for (float t : gpu_times) gpu_ms += t;
      gpu_ms /= static_cast<float>(gpu_times.size());
    }

    // 拷贝 GPU 结果回主机
    CHECK_CUDA(cudaMemcpy(h_Q_gpu.data(), gs.d_Q, h_Q_gpu.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_K_gpu.data(), gs.d_K, h_K_gpu.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_V_gpu.data(), gs.d_V, h_V_gpu.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // 误差校验
    const double max_diff_q = common::MaxAbsDiff(h_Q_cpu, h_Q_gpu);
    const double max_diff_k = common::MaxAbsDiff(h_K_cpu, h_K_gpu);
    const double max_diff_v = common::MaxAbsDiff(h_V_cpu, h_V_gpu);
    const bool ok = (max_diff_q < 1e-2 && max_diff_k < 1e-2 && max_diff_v < 1e-2);
    const char* check = ok ? "PASS" : "FAIL";

    // 清理 GPU 资源
    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    fused_v3::FreeGpuState(gs);

    // 输出结果
    const double speedup = (gpu_ms > 0) ? cpu_ms / gpu_ms : 0.0;
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

  std::cout << "\nResults saved to " << results_dir << "/fused_conv1d_silu_v3_results.csv\n";
  return 0;
}
