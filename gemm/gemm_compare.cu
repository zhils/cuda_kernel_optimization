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

struct ImplEntry {
  const char* label;
  gemm::ImplId impl;
  common::DType dtype_a;
  common::DType dtype_b;
  common::DType dtype_c;
  double tol;
};

ImplEntry g_impls[] = {
  {"cuBLAS-fp32",  gemm::ImplId::kCublas,      common::DType::kFp32,    common::DType::kFp32,    common::DType::kFp32, 1e-4},
  {"V3",           gemm::ImplId::kV3,           common::DType::kFp32,    common::DType::kFp32,    common::DType::kFp32, 1e-4},
  {"Fp16",         gemm::ImplId::kFp16,         common::DType::kFp16,    common::DType::kFp16,    common::DType::kFp32, 2.0},
  {"cuBLAS-fp16",  gemm::ImplId::kCublasFp16,   common::DType::kFp16,    common::DType::kFp16,    common::DType::kFp32, 2.0},
  {"cuBLAS-bf16",  gemm::ImplId::kCublasBf16,   common::DType::kBf16,    common::DType::kBf16,    common::DType::kFp32, 10.0},
  {"cuBLAS-int8",  gemm::ImplId::kCublasInt8,   common::DType::kInt8,    common::DType::kInt8,    common::DType::kFp32, 15.0},
  {"cuBLAS-fp8e4", gemm::ImplId::kCublasFp8,    common::DType::kFp8E4M3, common::DType::kFp8E4M3, common::DType::kFp32, 5.0},
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
  {"cuBLAS-fp8e5", gemm::ImplId::kCublasFp8,    common::DType::kFp8E5M2, common::DType::kFp8E5M2, common::DType::kFp32, 10.0},
#endif
};

constexpr int kNumImpls = sizeof(g_impls) / sizeof(g_impls[0]);

void CpuGemm(const float* A, const float* B, float* C,
             int M, int N, int K, int ldb) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0;
      for (int k = 0; k < K; ++k)
        sum += static_cast<double>(A[i * K + k]) * static_cast<double>(B[k * ldb + j]);
      C[i * N + j] = static_cast<float>(sum);
    }
  }
}

struct ImplResult {
  float gpu_ms = 0;
  double gflops = 0;
  bool ok = false;
  bool skipped = false;
  double max_diff = 0;
  bool check_pass = true;
};

ImplResult RunOneImpl(const gemm::GemmParams& p_template, int M, int N, int K,
                       const ImplEntry& ie, int ldb_alloc,
                       float* d_A, float* d_B, float* d_C,
                       const float* C_cpu, size_t n_c, bool do_cpu) {
  ImplResult r;

  gemm::GemmParams p = p_template;
  p.dtype_a = ie.dtype_a;
  p.dtype_b = ie.dtype_b;
  p.dtype_c = ie.dtype_c;
  p.impl = ie.impl;

  for (int w = 0; w < kWarmup; ++w) {
    auto st = gemm::GemmRun(p, nullptr);
    if (!st.ok()) {
      r.skipped = true;
      return r;
    }
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
      r.skipped = true;
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

  if (r.skipped || gpu_times.empty()) {
    r.skipped = true;
    return r;
  }

  std::sort(gpu_times.begin(), gpu_times.end());
  if (gpu_times.size() > 2) {
    for (size_t ti = 1; ti + 1 < gpu_times.size(); ++ti) r.gpu_ms += gpu_times[ti];
    r.gpu_ms /= static_cast<float>(gpu_times.size() - 2);
  } else {
    for (float t : gpu_times) r.gpu_ms += t;
    r.gpu_ms /= static_cast<float>(gpu_times.size());
  }

  const double flops = 2.0 * static_cast<double>(M) * N * K;
  r.gflops = (r.gpu_ms > 0) ? flops / (r.gpu_ms * 1e6) : 0;

  std::vector<float> C_gpu(n_c);
  CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, n_c * sizeof(float), cudaMemcpyDeviceToHost));

  if (do_cpu) {
    std::vector<float> C_cpu_vec(C_cpu, C_cpu + n_c);
    r.max_diff = common::MaxAbsDiff(C_cpu_vec, C_gpu);
    r.check_pass = (r.max_diff < ie.tol);
  }

  r.ok = true;
  return r;
}

}  // namespace

int main() {
  std::cout << "=== GEMM Head-to-Head Implementation Comparison ===\n\n";
  std::cout << "GPU: ";
  {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    std::cout << prop.name << "  (CC " << prop.major << "." << prop.minor << ")\n";
  }
  std::cout << "Baseline: cuBLAS fp32 (cublasLtMatmul, CUBLAS_COMPUTE_32F)\n\n";

  const std::vector<std::tuple<int, int, int, const char*>> sizes = {
      {128,   128,  128,  "[square]  128x128x128"},
      {256,   256,  256,  "[square]  256x256x256"},
      {512,   512,  512,  "[square]  512x512x512"},
      {1024,  1024, 1024, "[square]  1024x1024x1024"},
      {2048,  2048, 2048, "[square]  2048x2048x2048"},
      {4096,  4096, 4096, "[square]  4096x4096x4096"},
      {128,   96,   64,  "[tall]    128x96x64__QKV"},
      {256,   192,  128, "[tall]    256x192x128_QKV"},
      {1024,  384,  256, "[tall]    1024x384x256_QKV"},
      {4096,  768,  512, "[tall]    4096x768x512_QKV"},
      {128,   32,   64,  "[tall]    128x32x64___Z"},
      {1024,  128,  256, "[tall]    1024x128x256__Z"},
      {8192,  64,   128, "[xtall]   8192x64x128"},
      {64,    2048, 256, "[wide]    64x2048x256"},
      {128,   4096, 512, "[wide]    128x4096x512"},
      {1024,  1024, 64,  "[smallK]  1024x1024x64"},
      {64,    64,   4096,"[largeK]  64x64x4096"},
      {128,   256,  2048,"[largeK]  128x256x2048"},
      {100,   100,  100, "[unalign] 100x100x100"},
      {1023,  1025, 511, "[unalign] 1023x1025x511"},
      {1,     1,    1,   "[tiny]    1x1x1"},
      {1,     1,    128, "[tiny]    1x1x128"},
      {1,     128,  128, "[tiny]    1x128x128"},
      {128,   1,    128, "[tiny]    128x1x128"},
  };

  const int col_shape = 28;
  const int col_impl = 14;
  const int col_w = 12;
  const int col_sp = 16;
  std::cout << std::left
            << std::setw(col_shape) << "shape"
            << std::setw(8) << "M"
            << std::setw(8) << "N"
            << std::setw(8) << "K"
            << std::setw(col_impl) << "impl"
            << std::setw(8) << "dtype"
            << std::setw(col_w) << "GFLOPS"
            << std::setw(col_sp) << "vs cuBLAS fp32"
            << std::setw(9) << "Check" << "\n";
  std::cout << std::string(124, '-') << "\n";

  const std::string results_dir = common::EnsureResultsDir();
  std::ofstream ofs(results_dir + "/gemm_compare_results.csv");
  ofs << "shape,M,N,K,impl,dtype,gflops,speedup_vs_cublas_fp32,baseline_gflops,check\n";

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

    const bool do_cpu = (static_cast<size_t>(M) * N * K <= 1024LL * 1024 * 1024);
    if (do_cpu) CpuGemm(A_fp32.data(), B_fp32_pad.data(), C_cpu.data(), M, N, K, ldb_alloc);

    float* d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, n_a * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, n_b * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, n_c * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, A_fp32.data(), n_a * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B_fp32_pad.data(), n_b * sizeof(float), cudaMemcpyHostToDevice));

    gemm::GemmParams p_template;
    p_template.M = M; p_template.N = N; p_template.K = K;
    p_template.layout = common::Layout::kRowMajor;
    p_template.A = d_A; p_template.B = d_B; p_template.C = d_C;
    p_template.lda = K;
    p_template.ldb = ldb_alloc;
    p_template.ldc = N;

    ImplResult results[kNumImpls];

    int baseline_idx = 0;
    for (int ii = 0; ii < kNumImpls; ++ii) {
      if (g_impls[ii].impl == gemm::ImplId::kCublas &&
          g_impls[ii].dtype_a == common::DType::kFp32) {
        baseline_idx = ii;
        break;
      }
    }

    for (int ii = 0; ii < kNumImpls; ++ii) {
      results[ii] = RunOneImpl(p_template, M, N, K, g_impls[ii],
                                ldb_alloc, d_A, d_B, d_C,
                                C_cpu.data(), n_c, do_cpu);
    }

    const double baseline_gflops = (results[baseline_idx].ok && !results[baseline_idx].skipped)
                                      ? results[baseline_idx].gflops : 0;

    for (int ii = 0; ii < kNumImpls; ++ii) {
      const auto& ie = g_impls[ii];
      const auto& r = results[ii];

      std::cout << std::left << std::fixed
                << std::setw(col_shape) << shape_label
                << std::setw(8) << M
                << std::setw(8) << N
                << std::setw(8) << K
                << std::setw(col_impl) << ie.label;

      if (r.skipped) {
        std::cout << std::setw(8) << "SKIP"
                  << std::setw(col_w) << "-"
                  << std::setw(col_sp) << "-"
                  << std::setw(9) << "SKIP" << "\n";

        ofs << shape_label << "," << M << "," << N << "," << K << ","
            << ie.label << "," << "SKIP" << ","
            << "0" << "," << "0" << "," << "0" << "," << "SKIP" << "\n";
        continue;
      }

      const char* dtype_label = "fp32";
      if (ie.dtype_a == common::DType::kFp16) dtype_label = "fp16";
      else if (ie.dtype_a == common::DType::kBf16) dtype_label = "bf16";
      else if (ie.dtype_a == common::DType::kInt8) dtype_label = "int8";
      else if (ie.dtype_a == common::DType::kFp8E4M3) dtype_label = "fp8e4";
      else if (ie.dtype_a == common::DType::kFp8E5M2) dtype_label = "fp8e5";

      const double speedup = (baseline_gflops > 0) ? r.gflops / baseline_gflops : 0;

      std::cout << std::setw(8) << dtype_label
                << std::setprecision(1) << std::setw(col_w) << r.gflops
                << std::setw(col_sp);
      if (baseline_gflops > 0) {
        std::cout << std::setprecision(2) << speedup << "x";
      } else {
        std::cout << "-";
      }
      std::cout << std::setw(9) << (r.check_pass ? "PASS" : "FAIL") << "\n";

      ofs << shape_label << "," << M << "," << N << "," << K << ","
          << ie.label << "," << dtype_label << ","
          << r.gflops << "," << speedup << "," << baseline_gflops << ","
          << (r.check_pass ? "PASS" : "FAIL") << "\n";
    }

    std::cout << std::string(124, '-') << "\n";
    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
  }

  std::cout << "\nResults saved to " << results_dir << "/gemm_compare_results.csv\n";
  return 0;
}
