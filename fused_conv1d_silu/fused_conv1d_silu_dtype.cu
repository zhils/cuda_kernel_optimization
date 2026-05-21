// Fused Conv1D + SiLU 多 dtype 性能测试 — 使用融合 API 覆盖 fp32（基线）、bf16、int8、fp8_e4m3、fp8_e5m2。

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"
#include "fused/fused_api.h"
#include "fused/fused_ref_cpu.h"

namespace {

constexpr int kWarmup = 1;
constexpr int kRepeat = 10;

struct DTypeEntry {
  common::DType id;
  const char* name;
};

DTypeEntry g_dtypes[] = {
  {common::DType::kFp32,     "fp32"},
  {common::DType::kFp16,     "fp16"},
  {common::DType::kBf16,     "bf16"},
  {common::DType::kInt8,     "int8"},
  {common::DType::kFp8E4M3,  "fp8_e4m3"},
  {common::DType::kFp8E5M2,  "fp8_e5m2"},
};

constexpr int kNumDtypes = sizeof(g_dtypes) / sizeof(g_dtypes[0]);

struct GpuBuffers {
  float *d_x_fp32 = nullptr;
  float *d_W_qkv = nullptr;
  float *d_b_qkv = nullptr;
  float *d_W_z = nullptr;
  float *d_b_z = nullptr;
  float *d_K_conv = nullptr;

  float *d_Q = nullptr;
  float *d_K = nullptr;
  float *d_V = nullptr;

  __nv_bfloat16 *d_x_bf16 = nullptr;
  __nv_bfloat16 *d_Q_bf16 = nullptr;
  __nv_bfloat16 *d_K_bf16 = nullptr;
  __nv_bfloat16 *d_V_bf16 = nullptr;

  __half *d_x_f16 = nullptr;
  __half *d_Q_f16 = nullptr;
  __half *d_K_f16 = nullptr;
  __half *d_V_f16 = nullptr;

  int8_t *d_x_i8 = nullptr;
  int8_t *d_Q_i8 = nullptr;
  int8_t *d_K_i8 = nullptr;
  int8_t *d_V_i8 = nullptr;

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
  __nv_fp8_e4m3 *d_x_f8e4 = nullptr;
  __nv_fp8_e4m3 *d_Q_f8e4 = nullptr;
  __nv_fp8_e4m3 *d_K_f8e4 = nullptr;
  __nv_fp8_e4m3 *d_V_f8e4 = nullptr;

  __nv_fp8_e5m2 *d_x_f8e5 = nullptr;
  __nv_fp8_e5m2 *d_Q_f8e5 = nullptr;
  __nv_fp8_e5m2 *d_K_f8e5 = nullptr;
  __nv_fp8_e5m2 *d_V_f8e5 = nullptr;
#endif

  ~GpuBuffers() {
    cudaFree(d_x_fp32);
    cudaFree(d_W_qkv); cudaFree(d_b_qkv);
    cudaFree(d_W_z); cudaFree(d_b_z);
    cudaFree(d_K_conv);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_x_bf16);
    cudaFree(d_Q_bf16); cudaFree(d_K_bf16); cudaFree(d_V_bf16);
    cudaFree(d_x_f16);
    cudaFree(d_Q_f16); cudaFree(d_K_f16); cudaFree(d_V_f16);
    cudaFree(d_x_i8);
    cudaFree(d_Q_i8); cudaFree(d_K_i8); cudaFree(d_V_i8);
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
    cudaFree(d_x_f8e4);
    cudaFree(d_Q_f8e4); cudaFree(d_K_f8e4); cudaFree(d_V_f8e4);
    cudaFree(d_x_f8e5);
    cudaFree(d_Q_f8e5); cudaFree(d_K_f8e5); cudaFree(d_V_f8e5);
#endif
  }
};

void AllocGpu(GpuBuffers& g, int BL, int D, int H, int k_size) {
  size_t n_x = static_cast<size_t>(BL) * D;
  size_t n_out = static_cast<size_t>(BL) * H;

  CHECK_CUDA(cudaMalloc(&g.d_x_fp32, n_x * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&g.d_W_qkv, 3ULL * H * D * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&g.d_b_qkv, 3ULL * H * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&g.d_W_z, static_cast<size_t>(H) * D * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&g.d_b_z, H * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&g.d_K_conv, static_cast<size_t>(k_size) * H * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&g.d_Q, n_out * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&g.d_K, n_out * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&g.d_V, n_out * sizeof(float)));

  CHECK_CUDA(cudaMalloc(&g.d_x_bf16, n_x * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&g.d_Q_bf16, n_out * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&g.d_K_bf16, n_out * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&g.d_V_bf16, n_out * sizeof(__nv_bfloat16)));

  CHECK_CUDA(cudaMalloc(&g.d_x_f16, n_x * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&g.d_Q_f16, n_out * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&g.d_K_f16, n_out * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&g.d_V_f16, n_out * sizeof(__half)));

  CHECK_CUDA(cudaMalloc(&g.d_x_i8, n_x * sizeof(int8_t)));
  CHECK_CUDA(cudaMalloc(&g.d_Q_i8, n_out * sizeof(int8_t)));
  CHECK_CUDA(cudaMalloc(&g.d_K_i8, n_out * sizeof(int8_t)));
  CHECK_CUDA(cudaMalloc(&g.d_V_i8, n_out * sizeof(int8_t)));

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
  CHECK_CUDA(cudaMalloc(&g.d_x_f8e4, n_x * sizeof(__nv_fp8_e4m3)));
  CHECK_CUDA(cudaMalloc(&g.d_Q_f8e4, n_out * sizeof(__nv_fp8_e4m3)));
  CHECK_CUDA(cudaMalloc(&g.d_K_f8e4, n_out * sizeof(__nv_fp8_e4m3)));
  CHECK_CUDA(cudaMalloc(&g.d_V_f8e4, n_out * sizeof(__nv_fp8_e4m3)));

  CHECK_CUDA(cudaMalloc(&g.d_x_f8e5, n_x * sizeof(__nv_fp8_e5m2)));
  CHECK_CUDA(cudaMalloc(&g.d_Q_f8e5, n_out * sizeof(__nv_fp8_e5m2)));
  CHECK_CUDA(cudaMalloc(&g.d_K_f8e5, n_out * sizeof(__nv_fp8_e5m2)));
  CHECK_CUDA(cudaMalloc(&g.d_V_f8e5, n_out * sizeof(__nv_fp8_e5m2)));
#endif
}

void UploadWeights(GpuBuffers& g,
                   const std::vector<float>& W_qkv, const std::vector<float>& b_qkv,
                   const std::vector<float>& W_z, const std::vector<float>& b_z,
                   const std::vector<float>& K_conv) {
  CHECK_CUDA(cudaMemcpy(g.d_W_qkv, W_qkv.data(), W_qkv.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(g.d_b_qkv, b_qkv.data(), b_qkv.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(g.d_W_z, W_z.data(), W_z.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(g.d_b_z, b_z.data(), b_z.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(g.d_K_conv, K_conv.data(), K_conv.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
}

void PrepareInput(const std::vector<float>& x_fp32,
                  std::vector<__nv_bfloat16>& x_bf16,
                  std::vector<int8_t>& x_i8,
                  std::vector<float>& x_i8_fp32,
                  std::vector<__nv_fp8_e4m3>& x_f8e4,
                  std::vector<__nv_fp8_e5m2>& x_f8e5) {
  size_t n = x_fp32.size();
  x_bf16.resize(n);
  x_i8.resize(n);
  x_i8_fp32.resize(n);
  for (size_t i = 0; i < n; ++i) {
    x_bf16[i] = __float2bfloat16(x_fp32[i]);
    float r = roundf(x_fp32[i]);
    r = fminf(fmaxf(r, -128.0f), 127.0f);
    x_i8[i] = static_cast<int8_t>(r);
    x_i8_fp32[i] = static_cast<float>(x_i8[i]);
  }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
  x_f8e4.resize(n);
  x_f8e5.resize(n);
  for (size_t i = 0; i < n; ++i) {
    x_f8e4[i] = static_cast<__nv_fp8_e4m3>(x_fp32[i]);
    x_f8e5[i] = static_cast<__nv_fp8_e5m2>(x_fp32[i]);
  }
#endif
}

common::Status RunDtype(const fused::FusedParams& p, cudaStream_t stream) {
  return fused::FusedRun(p, stream);
}

bool IsDtypeSupported(common::DType dt) {
#if !(defined(CUDART_VERSION) && CUDART_VERSION >= 11080)
  if (dt == common::DType::kFp8E4M3 || dt == common::DType::kFp8E5M2) {
    return false;
  }
#endif
  return true;
}

}  // namespace

int main() {
  const std::vector<std::tuple<int, int, int, int, int>> test_cases = {
      {1, 1,   8,   4,   1},
      {1, 4,   16,  8,   2},
      {1, 16,  32,  16,  2},
      {1, 128, 64,  32,  4},
      {1, 256, 128, 64,  4},
      {2, 512, 256, 128, 4},
      {4, 1024, 512, 256, 4},
      {8, 2048, 512, 256, 4},
  };

  const std::string results_dir = common::EnsureResultsDir();
  std::ofstream ofs(results_dir + "/fused_conv1d_silu_dtype_results.csv");
  ofs << "B,L,D,H,k_size,dtype,cpu_ms,gpu_ms,speedup,"
         "max_abs_diff_q,max_abs_diff_k,max_abs_diff_v,check\n";

  std::cout << "=== Fused Conv1D + SiLU  Dtype Benchmark ===\n\n";

  const int col_w = 12;
  std::cout << std::left
            << std::setw(6) << "B"
            << std::setw(6) << "L"
            << std::setw(6) << "D"
            << std::setw(6) << "H"
            << std::setw(8) << "k_size"
            << std::setw(col_w) << "dtype"
            << std::setw(col_w) << "CPU ms"
            << std::setw(col_w) << "GPU ms"
            << std::setw(10) << "SpdUp"
            << std::setw(9) << "Check" << "\n";
  std::cout << std::string(85, '-') << "\n";

  for (const auto& tc : test_cases) {
    const int B = std::get<0>(tc);
    const int L = std::get<1>(tc);
    const int D = std::get<2>(tc);
    const int H = std::get<3>(tc);
    const int k_size = std::get<4>(tc);
    const int BL = B * L;
    const size_t n_x = static_cast<size_t>(BL) * D;
    const size_t n_out = static_cast<size_t>(BL) * H;

    std::vector<float> x_fp32(n_x);
    std::vector<float> W_qkv(3ULL * H * D), b_qkv(3ULL * H);
    std::vector<float> W_z(static_cast<size_t>(H) * D), b_z(H);
    std::vector<float> K_conv(static_cast<size_t>(k_size) * H);
    std::vector<float> Q_cpu(n_out), K_cpu(n_out), V_fp32_cpu(n_out);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto rand_fill = [&](std::vector<float>& v) {
      for (auto& val : v) val = dist(gen);
    };
    rand_fill(x_fp32);
    rand_fill(W_qkv); rand_fill(b_qkv);
    rand_fill(W_z); rand_fill(b_z);
    rand_fill(K_conv);

    fused::FusedParams ref_p;
    ref_p.B = B; ref_p.L = L; ref_p.D = D; ref_p.H = H; ref_p.k_size = k_size;
    ref_p.dtype = common::DType::kFp32;
    ref_p.x = x_fp32.data();
    ref_p.W_qkv = W_qkv.data(); ref_p.b_qkv = b_qkv.data();
    ref_p.W_z = W_z.data(); ref_p.b_z = b_z.data();
    ref_p.K_conv = K_conv.data();
    ref_p.Q = Q_cpu.data(); ref_p.K = K_cpu.data(); ref_p.V = V_fp32_cpu.data();

    const auto t0 = std::chrono::high_resolution_clock::now();
    if (!fused::FusedReferenceHost(ref_p).ok()) {
      std::cerr << "CPU reference failed for case B=" << B << "\n";
      return 1;
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<__nv_bfloat16> x_bf16, Q_bf16_host, K_bf16_host, V_bf16_host;
    std::vector<int8_t> x_i8;
    std::vector<float> x_i8_fp32;
    std::vector<__nv_fp8_e4m3> x_f8e4;
    std::vector<__nv_fp8_e5m2> x_f8e5;
    PrepareInput(x_fp32, x_bf16, x_i8, x_i8_fp32, x_f8e4, x_f8e5);

    GpuBuffers g;
    AllocGpu(g, BL, D, H, k_size);
    UploadWeights(g, W_qkv, b_qkv, W_z, b_z, K_conv);

    CHECK_CUDA(cudaMemcpy(g.d_x_fp32, x_fp32.data(), n_x * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(g.d_x_bf16, x_bf16.data(), n_x * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(g.d_x_i8, x_i8.data(), n_x * sizeof(int8_t),
                          cudaMemcpyHostToDevice));
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
    CHECK_CUDA(cudaMemcpy(g.d_x_f8e4, x_f8e4.data(), n_x * sizeof(__nv_fp8_e4m3),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(g.d_x_f8e5, x_f8e5.data(), n_x * sizeof(__nv_fp8_e5m2),
                          cudaMemcpyHostToDevice));
#endif

    for (int di = 0; di < kNumDtypes; ++di) {
      const common::DType dt = g_dtypes[di].id;
      const char* dname = g_dtypes[di].name;

      if (!IsDtypeSupported(dt)) {
        std::cout << std::left
                  << std::setw(6) << B << std::setw(6) << L
                  << std::setw(6) << D << std::setw(6) << H
                  << std::setw(8) << k_size
                  << std::setw(col_w) << dname
                  << std::setw(col_w) << std::fixed << std::setprecision(3)
                  << (di == 0 ? cpu_ms : 0.0)
                  << std::setw(col_w) << "SKIP"
                  << std::setw(10) << "-"
                  << std::setw(9) << "SKIP" << "\n";
        ofs << B << "," << L << "," << D << "," << H << "," << k_size << ","
            << dname << "," << (di == 0 ? cpu_ms : 0.0) << ",SKIP,0,0,0,0,SKIP\n";
        continue;
      }

      fused::FusedParams p;
      p.B = B; p.L = L; p.D = D; p.H = H; p.k_size = k_size;
      p.dtype = dt;
      p.impl = fused::ImplId::kAuto;
      p.W_qkv = g.d_W_qkv; p.b_qkv = g.d_b_qkv;
      p.W_z = g.d_W_z; p.b_z = g.d_b_z;
      p.K_conv = g.d_K_conv;

      switch (dt) {
        case common::DType::kFp32:
          p.x = g.d_x_fp32;
          p.Q = g.d_Q; p.K = g.d_K; p.V = g.d_V;
          break;
        case common::DType::kFp16:
          p.x = g.d_x_f16;
          p.Q = g.d_Q_f16; p.K = g.d_K_f16; p.V = g.d_V_f16;
          break;
        case common::DType::kBf16:
          p.x = g.d_x_bf16;
          p.Q = g.d_Q_bf16; p.K = g.d_K_bf16; p.V = g.d_V_bf16;
          break;
        case common::DType::kInt8:
          p.x = g.d_x_i8;
          p.Q = g.d_Q_i8; p.K = g.d_K_i8; p.V = g.d_V_i8;
          break;
        case common::DType::kFp8E4M3:
          p.x = g.d_x_f8e4;
          p.Q = g.d_Q_f8e4; p.K = g.d_K_f8e4; p.V = g.d_V_f8e4;
          break;
        case common::DType::kFp8E5M2:
          p.x = g.d_x_f8e5;
          p.Q = g.d_Q_f8e5; p.K = g.d_K_f8e5; p.V = g.d_V_f8e5;
          break;
        default:
          break;
      }

      {
        common::Status st = RunDtype(p, nullptr);
        CHECK_CUDA(cudaDeviceSynchronize());
      }

      for (int w = 0; w < kWarmup; ++w) {
        common::Status st = RunDtype(p, nullptr);
        (void)st;
      }
      CHECK_CUDA(cudaDeviceSynchronize());

      cudaEvent_t s, e;
      CHECK_CUDA(cudaEventCreate(&s));
      CHECK_CUDA(cudaEventCreate(&e));

      std::vector<float> gpu_times;
      for (int rep = 0; rep < kRepeat; ++rep) {
        CHECK_CUDA(cudaEventRecord(s));
        common::Status st = RunDtype(p, nullptr);
        (void)st;
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
        for (size_t i = 1; i + 1 < gpu_times.size(); ++i) gpu_ms += gpu_times[i];
        gpu_ms /= static_cast<float>(gpu_times.size() - 2);
      } else {
        for (float t : gpu_times) gpu_ms += t;
        gpu_ms /= static_cast<float>(gpu_times.size());
      }

      double max_diff_q = 0.0, max_diff_k = 0.0, max_diff_v = 0.0;
      bool ok = true;

      std::vector<float> Q_gpu(n_out), K_gpu(n_out), V_gpu(n_out);

      switch (dt) {
        case common::DType::kFp32:
          CHECK_CUDA(cudaMemcpy(Q_gpu.data(), g.d_Q, n_out * sizeof(float), cudaMemcpyDeviceToHost));
          CHECK_CUDA(cudaMemcpy(K_gpu.data(), g.d_K, n_out * sizeof(float), cudaMemcpyDeviceToHost));
          CHECK_CUDA(cudaMemcpy(V_gpu.data(), g.d_V, n_out * sizeof(float), cudaMemcpyDeviceToHost));
          break;
        case common::DType::kFp16: {
          std::vector<__half> tmp(n_out);
          CHECK_CUDA(cudaMemcpy(tmp.data(), g.d_Q_f16, n_out * sizeof(__half), cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < n_out; ++i) Q_gpu[i] = __half2float(tmp[i]);
          CHECK_CUDA(cudaMemcpy(tmp.data(), g.d_K_f16, n_out * sizeof(__half), cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < n_out; ++i) K_gpu[i] = __half2float(tmp[i]);
          CHECK_CUDA(cudaMemcpy(tmp.data(), g.d_V_f16, n_out * sizeof(__half), cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < n_out; ++i) V_gpu[i] = __half2float(tmp[i]);
          break;
        }
        case common::DType::kBf16: {
          std::vector<__nv_bfloat16> tmp(n_out);
          CHECK_CUDA(cudaMemcpy(tmp.data(), g.d_Q_bf16, n_out * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < n_out; ++i) Q_gpu[i] = __bfloat162float(tmp[i]);
          CHECK_CUDA(cudaMemcpy(tmp.data(), g.d_K_bf16, n_out * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < n_out; ++i) K_gpu[i] = __bfloat162float(tmp[i]);
          CHECK_CUDA(cudaMemcpy(tmp.data(), g.d_V_bf16, n_out * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < n_out; ++i) V_gpu[i] = __bfloat162float(tmp[i]);
          break;
        }
        case common::DType::kInt8: {
          std::vector<int8_t> tmp(n_out);
          CHECK_CUDA(cudaMemcpy(tmp.data(), g.d_Q_i8, n_out * sizeof(int8_t), cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < n_out; ++i) Q_gpu[i] = static_cast<float>(tmp[i]);
          CHECK_CUDA(cudaMemcpy(tmp.data(), g.d_K_i8, n_out * sizeof(int8_t), cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < n_out; ++i) K_gpu[i] = static_cast<float>(tmp[i]);
          CHECK_CUDA(cudaMemcpy(tmp.data(), g.d_V_i8, n_out * sizeof(int8_t), cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < n_out; ++i) V_gpu[i] = static_cast<float>(tmp[i]);
          break;
        }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
        case common::DType::kFp8E4M3: {
          std::vector<__nv_fp8_e4m3> tmp(n_out);
          CHECK_CUDA(cudaMemcpy(tmp.data(), g.d_Q_f8e4, n_out * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < n_out; ++i) Q_gpu[i] = static_cast<float>(tmp[i]);
          CHECK_CUDA(cudaMemcpy(tmp.data(), g.d_K_f8e4, n_out * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < n_out; ++i) K_gpu[i] = static_cast<float>(tmp[i]);
          CHECK_CUDA(cudaMemcpy(tmp.data(), g.d_V_f8e4, n_out * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < n_out; ++i) V_gpu[i] = static_cast<float>(tmp[i]);
          break;
        }
        case common::DType::kFp8E5M2: {
          std::vector<__nv_fp8_e5m2> tmp(n_out);
          CHECK_CUDA(cudaMemcpy(tmp.data(), g.d_Q_f8e5, n_out * sizeof(__nv_fp8_e5m2), cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < n_out; ++i) Q_gpu[i] = static_cast<float>(tmp[i]);
          CHECK_CUDA(cudaMemcpy(tmp.data(), g.d_K_f8e5, n_out * sizeof(__nv_fp8_e5m2), cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < n_out; ++i) K_gpu[i] = static_cast<float>(tmp[i]);
          CHECK_CUDA(cudaMemcpy(tmp.data(), g.d_V_f8e5, n_out * sizeof(__nv_fp8_e5m2), cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < n_out; ++i) V_gpu[i] = static_cast<float>(tmp[i]);
          break;
        }
#endif
        default:
          break;
      }

      max_diff_q = common::MaxAbsDiff(Q_cpu, Q_gpu);
      max_diff_k = common::MaxAbsDiff(K_cpu, K_gpu);
      max_diff_v = common::MaxAbsDiff(V_fp32_cpu, V_gpu);

      float tol = 1e-2f;
      if (dt == common::DType::kFp16) tol = 0.5f;
      else if (dt == common::DType::kInt8) tol = 5.0f;
      else if (dt == common::DType::kFp8E4M3) tol = 1.0f;
      else if (dt == common::DType::kFp8E5M2) tol = 1.0f;
      ok = (max_diff_q < tol && max_diff_k < tol && max_diff_v < tol);
      const char* check = ok ? "PASS" : "FAIL";

      const double speedup = (gpu_ms > 0) ? cpu_ms / gpu_ms : 0.0;

      std::cout << std::left
                << std::setw(6) << B << std::setw(6) << L
                << std::setw(6) << D << std::setw(6) << H
                << std::setw(8) << k_size
                << std::setw(col_w) << dname
                << std::fixed << std::setprecision(3)
                << std::setw(col_w) << (di == 0 ? cpu_ms : 0.0)
                << std::setw(col_w) << gpu_ms
                << std::setw(10) << std::setprecision(2) << speedup
                << std::setw(9) << check << "\n";

      ofs << B << "," << L << "," << D << "," << H << "," << k_size << ","
          << dname << "," << (di == 0 ? cpu_ms : 0.0) << "," << gpu_ms << ","
          << speedup << ","
          << max_diff_q << "," << max_diff_k << "," << max_diff_v << ","
          << check << "\n";

      CHECK_CUDA(cudaEventDestroy(s));
      CHECK_CUDA(cudaEventDestroy(e));
    }  // dtype loop

    std::cout << std::string(85, '-') << "\n";
  }

  std::cout << "\nResults saved to " << results_dir
            << "/fused_conv1d_silu_dtype_results.csv\n";
  return 0;
}
