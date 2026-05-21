#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

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
#include "gemm/gemm_quant.h"

namespace {

constexpr int kWarmup = 3;
constexpr int kRepeat = 10;

void CpuGemm(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < N; ++c) {
      float s = 0;
      for (int k = 0; k < K; ++k) s += A[r * K + k] * B[k * N + c];
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
  std::cout << "=== GEMM Quantized Benchmark (fp32→quantize→GEMM→dequantize→fp32) ===\n\n";

  const std::string results_dir = common::EnsureResultsDir();
  std::ofstream ofs(results_dir + "/gemm_quant_results.csv");
  ofs << "shape,M,N,K,dtype,scheme,cpu_ms,gpu_ms,speedup,max_diff,check\n";

  const int col_w = 14;
  const int col_shape = 30;
  std::cout << std::left
            << std::setw(col_shape) << "shape"
            << std::setw(8) << "M" << std::setw(8) << "N" << std::setw(8) << "K"
            << std::setw(col_w) << "dtype"
            << std::setw(col_w) << "scheme"
            << std::setw(col_w) << "GPU ms"
            << std::setw(10) << "speedup"
            << std::setw(12) << "max_diff"
            << std::setw(8) << "check\n";
  std::cout << std::string(118, '-') << "\n";

  const std::vector<std::tuple<int, int, int, const char*>> sizes = {
      {128,   128,  128,  "[square] M128_N128_K128"},
      {256,   256,  256,  "[square] M256_N256_K256"},
      {512,   512,  512,  "[square] M512_N512_K512"},
      {1024,  1024, 1024, "[square] M1K_N1K_K1K"},
      {2048,  2048, 2048, "[square] M2K_N2K_K2K"},
      {4096,  4096, 4096, "[square] M4K_N4K_K4K"},
      {4096,  768,  512,  "[tall]   M4K_N768_K512"},
      {1024,  128,  256,  "[tall]   M1K_N128_K256"},
      {100,   100,  100,  "[unalign] M100_N100_K100"},
      {1023,  1025, 511,  "[unalign] M1023_N1025_K511"},
  };

  struct Config {
    common::DType dtype;
    const char* label;
    gemm::QuantScheme scheme_a;
    gemm::QuantScheme scheme_b;
    const char* s_label;
  };

  Config configs[] = {
    {common::DType::kInt8,     "int8",     gemm::QuantScheme::kPerTensor, gemm::QuantScheme::kPerTensor, "per_tensor"},
    {common::DType::kInt8,     "int8",     gemm::QuantScheme::kPerRow,   gemm::QuantScheme::kPerTensor, "per_row_A"},
    {common::DType::kFp8E4M3,  "fp8_e4m3", gemm::QuantScheme::kPerTensor, gemm::QuantScheme::kPerTensor, "per_tensor"},
    {common::DType::kFp8E4M3,  "fp8_e4m3", gemm::QuantScheme::kPerRow,   gemm::QuantScheme::kPerTensor, "per_row_A"},
  };

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  for (const auto& sz : sizes) {
    int M = std::get<0>(sz), N = std::get<1>(sz), K = std::get<2>(sz);
    const char* shape_label = std::get<3>(sz);
    const size_t n_a = static_cast<size_t>(M) * K;
    const size_t n_b = static_cast<size_t>(K) * N;
    const size_t n_c = static_cast<size_t>(M) * N;

    std::vector<float> A_fp32(n_a), B_fp32(n_b), C_ref(n_c);
    for (auto& v : A_fp32) v = dist(gen);
    for (auto& v : B_fp32) v = dist(gen);

    double cpu_ms = 0;
    const bool do_cpu = (n_c * K <= 512ULL * 1024 * 1024);
    if (do_cpu) {
      const auto t0 = std::chrono::high_resolution_clock::now();
      CpuGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);
      const auto t1 = std::chrono::high_resolution_clock::now();
      cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C_baseline = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, n_a * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, n_b * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_baseline, n_c * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, A_fp32.data(), n_a * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B_fp32.data(), n_b * sizeof(float), cudaMemcpyHostToDevice));

    float baseline_ms = 0;
    {
      gemm::GemmParams p;
      p.M = M; p.N = N; p.K = K;
      p.dtype_a = common::DType::kFp32;
      p.dtype_b = common::DType::kFp32;
      p.dtype_c = common::DType::kFp32;
      p.layout = common::Layout::kRowMajor;
      p.A = d_A; p.B = d_B; p.C = d_C_baseline;
      p.lda = K; p.ldb = N; p.ldc = N;
      p.impl = gemm::ImplId::kAuto;

      for (int w = 0; w < kWarmup; ++w) gemm::GemmRun(p, nullptr);
      CHECK_CUDA(cudaDeviceSynchronize());

      cudaEvent_t s, e;
      CHECK_CUDA(cudaEventCreate(&s));
      CHECK_CUDA(cudaEventCreate(&e));
      std::vector<float> times;
      for (int rep = 0; rep < kRepeat; ++rep) {
        CHECK_CUDA(cudaEventRecord(s));
        gemm::GemmRun(p, nullptr);
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
        times.push_back(ms);
      }
      CHECK_CUDA(cudaEventDestroy(s));
      CHECK_CUDA(cudaEventDestroy(e));
      std::sort(times.begin(), times.end());
      if (times.size() > 2) {
        for (size_t ti = 1; ti + 1 < times.size(); ++ti) baseline_ms += times[ti];
        baseline_ms /= static_cast<float>(times.size() - 2);
      } else {
        for (float t : times) baseline_ms += t;
        baseline_ms /= static_cast<float>(times.size());
      }
    }

    std::cout << std::left
              << std::setw(col_shape) << shape_label
              << std::setw(8) << M << std::setw(8) << N << std::setw(8) << K
              << std::setw(col_w) << "fp32"
              << std::setw(col_w) << "baseline"
              << std::setw(col_w) << std::fixed << std::setprecision(4) << baseline_ms
              << std::setw(10) << "1.00x"
              << std::setw(12) << "-"
              << std::setw(8) << "-\n";

    float* d_C_quant = nullptr;
    CHECK_CUDA(cudaMalloc(&d_C_quant, n_c * sizeof(float)));

    for (const auto& cfg : configs) {
      if (!DTypeSupported(cfg.dtype)) {
        std::cout << std::left
                  << std::setw(col_shape) << shape_label
                  << std::setw(8) << M << std::setw(8) << N << std::setw(8) << K
                  << std::setw(col_w) << cfg.label
                  << std::setw(col_w) << cfg.s_label
                  << std::setw(col_w) << "SKIP"
                  << std::setw(10) << "-"
                  << std::setw(12) << "-"
                  << std::setw(8) << "-\n";
        continue;
      }

      for (int w = 0; w < kWarmup; ++w) {
        gemm::GemmQuantizedRun(d_A, d_B, d_C_quant, M, N, K,
                               cfg.dtype, cfg.scheme_a, cfg.scheme_b, nullptr);
      }
      CHECK_CUDA(cudaDeviceSynchronize());

      cudaEvent_t s, e;
      CHECK_CUDA(cudaEventCreate(&s));
      CHECK_CUDA(cudaEventCreate(&e));
      std::vector<float> gpu_times;
      for (int rep = 0; rep < kRepeat; ++rep) {
        CHECK_CUDA(cudaEventRecord(s));
        auto st = gemm::GemmQuantizedRun(d_A, d_B, d_C_quant, M, N, K,
                                         cfg.dtype, cfg.scheme_a, cfg.scheme_b, nullptr);
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));
        if (!st.ok()) {
          std::cerr << "QuantGEMM " << cfg.label << " FAIL: " << st.message << "\n";
          break;
        }
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

      double speedup = (baseline_ms > 0 && gpu_ms > 0) ? baseline_ms / gpu_ms : 0;

      std::vector<float> C_gpu(n_c);
      CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C_quant, n_c * sizeof(float), cudaMemcpyDeviceToHost));

      double max_diff = 0;
      std::string check_str = "-";
      if (do_cpu) {
        max_diff = common::MaxAbsDiff(C_ref, C_gpu);
        double tol = 1e-4;
        if (cfg.dtype == common::DType::kInt8) tol = 5.0;
        else if (cfg.dtype == common::DType::kFp8E4M3) tol = 3.0;
        check_str = (max_diff < tol) ? "PASS" : "FAIL";
      }

      std::cout << std::left
                << std::setw(col_shape) << shape_label
                << std::setw(8) << M << std::setw(8) << N << std::setw(8) << K
                << std::setw(col_w) << cfg.label
                << std::setw(col_w) << cfg.s_label
                << std::fixed << std::setprecision(4)
                << std::setw(col_w) << gpu_ms
                << std::setw(10) << std::setprecision(2) << speedup << "x"
                << std::setw(12) << std::setprecision(4) << max_diff
                << std::setw(8) << check_str << "\n";

      ofs << shape_label << "," << M << "," << N << "," << K << ","
          << cfg.label << "," << cfg.s_label << ","
          << (do_cpu ? cpu_ms : 0.0) << "," << gpu_ms << ","
          << speedup << "," << max_diff << "," << check_str << "\n";
    }

    CHECK_CUDA(cudaFree(d_C_quant));
    std::cout << std::string(118, '-') << "\n";

    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C_baseline));
  }

  std::cout << "\nResults saved to " << results_dir << "/gemm_quant_results.csv\n";
  return 0;
}
