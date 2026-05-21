#include <cuda_runtime.h>
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
#include "gemm/gemm_api.h"

namespace {

constexpr int kWarmup = 3;
constexpr int kRepeat = 10;

#define CPU_TOL 1

struct DTypeId {
  common::DType a, b, c;
  const char* label;
};

DTypeId g_specs[] = {
  {common::DType::kFp32,     common::DType::kFp32,     common::DType::kFp32,  "fp32"},
  {common::DType::kFp16,     common::DType::kFp16,     common::DType::kFp32,  "fp16"},
  {common::DType::kBf16,     common::DType::kBf16,     common::DType::kFp32,  "bf16"},
  {common::DType::kInt8,     common::DType::kInt8,     common::DType::kFp32,  "int8"},
  {common::DType::kFp8E4M3,  common::DType::kFp8E4M3,  common::DType::kFp32,  "fp8_e4m3"},
  {common::DType::kFp8E5M2,  common::DType::kFp8E5M2,  common::DType::kFp32,  "fp8_e5m2"},
};
constexpr int kNumSpecs = sizeof(g_specs) / sizeof(g_specs[0]);

void CpuGemm(const float* A, const float* B, float* C, int M, int N, int K, int ldb = 0) {
  if (ldb <= 0) ldb = N;
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < N; ++c) {
      float s = 0;
      for (int k = 0; k < K; ++k) s += A[r * K + k] * B[k * ldb + c];
      C[r * N + c] = s;
    }
}

bool DTypeSupported(common::DType dt) {
#if !(defined(CUDART_VERSION) && CUDART_VERSION >= 11080)
  if (dt == common::DType::kFp8E4M3 || dt == common::DType::kFp8E5M2) return false;
#endif
  return true;
}

}  // namespace

int main() {
  std::cout << "=== GEMM Dtype Benchmark (gemm::GemmRun) ===\n\n";

  const std::string results_dir = common::EnsureResultsDir();
  std::ofstream ofs(results_dir + "/gemm_dtype_results.csv");
  ofs << "shape,M,N,K,dtype,cpu_ms,gpu_ms,gflops,max_abs_diff,check\n";

  const int col_w = 12;
  const int col_shape = 33;
  std::cout << std::left
            << std::setw(col_shape) << "shape"
            << std::setw(7) << "M"
            << std::setw(7) << "N"
            << std::setw(7) << "K"
            << std::setw(col_w) << "dtype"
            << std::setw(col_w) << "CPU ms"
            << std::setw(col_w) << "GPU ms"
            << std::setw(10) << "GFLOPs"
            << std::setw(9) << "Check" << "\n";
  std::cout << std::string(111, '-') << "\n";

  const std::vector<std::tuple<int, int, int, const char*>> sizes = {
      {1,     1,    1,    "[tiny]    M1_N1_K1"},
      {1,     1,    128,  "[tiny]    M1_N1_K128"},
      {1,     128,  128,  "[tiny]    M1_N128_K128"},
      {128,   1,    128,  "[tiny]    M128_N1_K128"},
      {8,     8,    8,    "[tiny]    M8_N8_K8"},
      {128,   128,  128,  "[square]  M128_N128_K128"},
      {256,   256,  256,  "[square]  M256_N256_K256"},
      {512,   512,  512,  "[square]  M512_N512_K512"},
      {1024,  1024, 1024, "[square]  M1K_N1K_K1K"},
      {2048,  2048, 2048, "[square]  M2K_N2K_K2K"},
      {4096,  4096, 4096, "[square]  M4K_N4K_K4K"},
      {128,   96,   64,  "[tall]    M128_N96_K64__QKV"},
      {256,   192,  128, "[tall]    M256_N192_K128_QKV"},
      {1024,  384,  256, "[tall]    M1K_N384_K256_QKV"},
      {4096,  768,  512, "[tall]    M4K_N768_K512_QKV"},
      {128,   32,   64,  "[tall]    M128_N32_K64___Z"},
      {1024,  128,  256, "[tall]    M1K_N128_K256__Z"},
      {4096,  256,  512, "[tall]    M4K_N256_K512__Z"},
      {8192,  64,   128, "[xtall]   M8K_N64_K128"},
      {16384, 96,   64,  "[xtall]   M16K_N96_K64"},
      {64,    2048, 256, "[wide]    M64_N2K_K256"},
      {128,   4096, 512, "[wide]    M128_N4K_K512"},
      {1024,  1024, 64,  "[smallK]  M1K_N1K_K64"},
      {4096,  1024, 128, "[smallK]  M4K_N1K_K128"},
      {64,    64,   4096, "[largeK]  M64_N64_K4K"},
      {128,   256,  2048, "[largeK]  M128_N256_K2K"},
      {100,   100,  100, "[unalign] M100_N100_K100"},
      {1023,  1025, 511, "[unalign] M1023_N1025_K511"},
      {300,   250,  200, "[unalign] M300_N250_K200"},
  };

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  for (const auto& sz : sizes) {
    int M = std::get<0>(sz), N = std::get<1>(sz), K = std::get<2>(sz);
    const char* shape_label = std::get<3>(sz);
    const size_t n_a = static_cast<size_t>(M) * K;
    const int ldb_alloc = std::max(K, N);
    const size_t n_b = static_cast<size_t>(K) * ldb_alloc;
    const size_t n_c = static_cast<size_t>(M) * N;

    std::vector<float> A_fp32(n_a), B_fp32_pad(n_b), C_cpu(n_c);
    for (auto& v : A_fp32) v = dist(gen);
    for (size_t row = 0; row < static_cast<size_t>(K); ++row)
      for (int col = 0; col < N; ++col)
        B_fp32_pad[row * ldb_alloc + col] = dist(gen);

    double cpu_ms = 0;
    const bool do_cpu = (static_cast<size_t>(M) * N * K <= 1024LL * 1024 * 1024);
    if (do_cpu) {
      const auto t0 = std::chrono::high_resolution_clock::now();
      CpuGemm(A_fp32.data(), B_fp32_pad.data(), C_cpu.data(), M, N, K, ldb_alloc);
      const auto t1 = std::chrono::high_resolution_clock::now();
      cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    float* d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, n_a * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, n_b * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, n_c * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, A_fp32.data(), n_a * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B_fp32_pad.data(), n_b * sizeof(float), cudaMemcpyHostToDevice));

    for (int si = 0; si < kNumSpecs; ++si) {
      common::DType da = g_specs[si].a, db = g_specs[si].b, dc = g_specs[si].c;
      const char* label = g_specs[si].label;

      if (!DTypeSupported(da)) {
        std::cout << std::left
                  << std::setw(col_shape) << shape_label
                  << std::setw(7) << M << std::setw(7) << N << std::setw(7) << K
                  << std::setw(col_w) << label
                  << std::setw(col_w) << (si == 0 && do_cpu ? cpu_ms : 0.0)
                  << std::setw(col_w) << "SKIP"
                  << std::setw(10) << "-"
                  << std::setw(9) << "SKIP" << "\n";
        continue;
      }

      gemm::GemmParams p;
      p.M = M; p.N = N; p.K = K;
      p.dtype_a = da; p.dtype_b = db; p.dtype_c = dc;
      p.layout = common::Layout::kRowMajor;
      p.A = d_A; p.B = d_B; p.C = d_C;
      p.lda = K;
      p.ldb = ldb_alloc;
      p.ldc = N;
      p.impl = gemm::ImplId::kAuto;

      for (int w = 0; w < kWarmup; ++w) {
        gemm::GemmRun(p, nullptr);
      }
      CHECK_CUDA(cudaDeviceSynchronize());

      cudaEvent_t s, e;
      CHECK_CUDA(cudaEventCreate(&s));
      CHECK_CUDA(cudaEventCreate(&e));
      std::vector<float> gpu_times;
      for (int rep = 0; rep < kRepeat; ++rep) {
        CHECK_CUDA(cudaEventRecord(s));
        auto st = gemm::GemmRun(p, nullptr);
        if (!st.ok()) {
          std::cerr << "GEMM " << label << " FAIL: " << st.message << "\n";
          break;
        }
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));
        CHECK_CUDA(cudaGetLastError());
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
        gpu_times.push_back(ms);
      }
      CHECK_CUDA(cudaEventDestroy(s));
      CHECK_CUDA(cudaEventDestroy(e));

      if (gpu_times.empty()) continue;
      std::sort(gpu_times.begin(), gpu_times.end());
      float gpu_ms = 0;
      if (gpu_times.size() > 2) {
        for (size_t ti = 1; ti + 1 < gpu_times.size(); ++ti) gpu_ms += gpu_times[ti];
        gpu_ms /= static_cast<float>(gpu_times.size() - 2);
      } else {
        for (float t : gpu_times) gpu_ms += t;
        gpu_ms /= static_cast<float>(gpu_times.size());
      }

      const double flops = 2.0 * M * N * K;
      const double gflops = (gpu_ms > 0) ? flops / (gpu_ms * 1e6) : 0;

      std::vector<float> C_gpu(n_c);
      CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, n_c * sizeof(float), cudaMemcpyDeviceToHost));

      double max_diff = 0;
      bool check_ok = true;
      if (do_cpu) {
        max_diff = common::MaxAbsDiff(C_cpu, C_gpu);
        double tol = 1e-4;
        if (da == common::DType::kInt8) tol = 15.0;
        else if (da == common::DType::kFp8E4M3) tol = 5.0;
        else if (da == common::DType::kFp8E5M2) tol = 10.0;
        else if (da == common::DType::kFp16) tol = 2.0;
        else if (da == common::DType::kBf16) tol = 10.0;
        check_ok = (max_diff < tol);
      }

      std::cout << std::left
                << std::setw(col_shape) << shape_label
                << std::setw(7) << M << std::setw(7) << N << std::setw(7) << K
                << std::setw(col_w) << label
                << std::fixed << std::setprecision(3)
                << std::setw(col_w) << (si == 0 && do_cpu ? cpu_ms : 0.0)
                << std::setw(col_w) << gpu_ms
                << std::setw(10) << std::setprecision(1) << gflops
                << std::setw(9) << (check_ok ? "PASS" : "FAIL") << "\n";

      ofs << shape_label << "," << M << "," << N << "," << K << "," << label << ","
          << (si == 0 && do_cpu ? cpu_ms : 0.0) << "," << gpu_ms << ","
          << gflops << "," << max_diff << "," << (check_ok ? "PASS" : "FAIL") << "\n";
    }

    std::cout << std::string(111, '-') << "\n";
    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
  }

  std::cout << "\nResults saved to " << results_dir << "/gemm_dtype_results.csv\n";
  return 0;
}
