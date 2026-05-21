#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "gemm_cute_fp16.cuh"

struct PrecisionMetrics {
  double cos_sim = 0.0;
  double snr_db = 0.0;
  double max_rel_err = 0.0;
  double mean_abs_err = 0.0;
  double p99_abs_err = 0.0;
  double max_abs_err = 0.0;
};

static PrecisionMetrics ComputeMetrics(const float* ref, const float* test, size_t n) {
  PrecisionMetrics m;
  double dot = 0.0, norm_ref = 0.0, norm_test = 0.0;
  double noise_power = 0.0, signal_power = 0.0;
  double max_rel = 0.0;
  std::vector<double> abs_errs(n);
  double sum_abs = 0.0;
  double max_abs = 0.0;

  double ref_max_abs = 0.0;
  for (size_t i = 0; i < n; ++i) {
    ref_max_abs = std::max(ref_max_abs, static_cast<double>(std::fabs(ref[i])));
  }
  const double rel_threshold = std::max(ref_max_abs * 1e-6, 1e-6);

  for (size_t i = 0; i < n; ++i) {
    const double r = ref[i];
    const double t = test[i];
    dot += r * t;
    norm_ref += r * r;
    norm_test += t * t;
    const double diff = r - t;
    noise_power += diff * diff;
    signal_power += r * r;
    const double ab = std::fabs(diff);
    abs_errs[i] = ab;
    sum_abs += ab;
    max_abs = std::max(max_abs, ab);
    if (std::fabs(r) >= rel_threshold) {
      max_rel = std::max(max_rel, ab / std::fabs(r));
    }
  }

  m.cos_sim = dot / (std::sqrt(norm_ref) * std::sqrt(norm_test) + 1e-10);
  m.snr_db = 10.0 * std::log10(signal_power / (noise_power + 1e-10));
  m.max_rel_err = max_rel;
  m.mean_abs_err = sum_abs / static_cast<double>(n);
  m.max_abs_err = max_abs;
  if (n > 0) {
    size_t p99_idx = static_cast<size_t>(n * 0.99);
    p99_idx = std::min(p99_idx, n - 1);
    std::nth_element(abs_errs.begin(), abs_errs.begin() + static_cast<std::ptrdiff_t>(p99_idx),
                     abs_errs.end());
    m.p99_abs_err = abs_errs[p99_idx];
  }
  return m;
}

static void GemmCPU_FP32(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c) {
      float s = 0.f;
      for (int k = 0; k < K; ++k) {
        s += A[static_cast<size_t>(r) * K + k] * B[static_cast<size_t>(k) * N + c];
      }
      C[static_cast<size_t>(r) * N + c] = s;
    }
  }
}

int main() {
  int major = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
  if (major < 8) {
    std::cout << "gemm_cute_fp16 requires SM80+ (Ampere or newer)\n";
    return 0;
  }

  constexpr int kWarmup = 3;
  constexpr int kRepeat = 10;
  constexpr int kMaxCpuVerifyDim = 1024;
  auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/gemm_cute_fp16_results.csv");
  ofs << "id,group,M,N,K,gpu_ms,gflops,"
      << "cos_sim,snr_db,max_rel_err,mean_abs_err,p99_abs_err,max_abs_err\n";

  std::cout << "CUTLASS FP16 GEMM (Sm80 Tensor Core, RowMajor, F32 accum)\n";
  std::cout << "Tile: " << gemm_cute::kTileM << "x" << gemm_cute::kTileN << "x"
            << gemm_cute::kTileK << "\n\n";

  for (size_t i = 0; i < cases.size(); ++i) {
    const int M = cases[i].rows;
    const int N = cases[i].cols;
    const int K = M;
    const bool aligned = gemm_cute::IsAligned(M, N, K);
    const size_t C_size = static_cast<size_t>(M) * N;
    std::vector<float> A_fp32(static_cast<size_t>(M) * K);
    std::vector<float> B_fp32(static_cast<size_t>(K) * N);
    std::vector<float> C_cpu(C_size);
    std::vector<float> C_gpu_fp32(C_size);
    common::InitMatrix(A_fp32, M, K);
    common::InitMatrix(B_fp32, K, N);

    if (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim) {
      GemmCPU_FP32(A_fp32.data(), B_fp32.data(), C_cpu.data(), M, N, K);
    }

    float gpu_ms = 0.f;
    PrecisionMetrics pm;
    if (aligned) {
      std::vector<cutlass::half_t> A_half(A_fp32.size());
      std::vector<cutlass::half_t> B_half(B_fp32.size());
      for (size_t j = 0; j < A_fp32.size(); ++j) {
        A_half[j] = cutlass::half_t(A_fp32[j]);
      }
      for (size_t j = 0; j < B_fp32.size(); ++j) {
        B_half[j] = cutlass::half_t(B_fp32[j]);
      }

      cutlass::half_t *dA = nullptr, *dB = nullptr;
      float* dC = nullptr;
      CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dA), A_half.size() * sizeof(cutlass::half_t)));
      CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dB), B_half.size() * sizeof(cutlass::half_t)));
      CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dC), C_size * sizeof(float)));
      CHECK_CUDA(cudaMemcpy(dA, A_half.data(), A_half.size() * sizeof(cutlass::half_t),
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(dB, B_half.data(), B_half.size() * sizeof(cutlass::half_t),
                            cudaMemcpyHostToDevice));

      for (int w = 0; w < kWarmup; ++w) {
        if (gemm_cute::LaunchGemmFp16RowMajor(M, N, K, dA, dB, dC, 1.f, 0.f, nullptr) !=
            cutlass::Status::kSuccess) {
          std::cerr << "Launch failed at warmup\n";
          break;
        }
      }
      CHECK_CUDA(cudaDeviceSynchronize());

      cudaEvent_t start, stop;
      CHECK_CUDA(cudaEventCreate(&start));
      CHECK_CUDA(cudaEventCreate(&stop));
      CHECK_CUDA(cudaEventRecord(start));
      for (int rep = 0; rep < kRepeat; ++rep) {
        gemm_cute::LaunchGemmFp16RowMajor(M, N, K, dA, dB, dC, 1.f, 0.f, nullptr);
      }
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));
      gpu_ms /= static_cast<float>(kRepeat);

      CHECK_CUDA(cudaMemcpy(C_gpu_fp32.data(), dC, C_size * sizeof(float), cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaEventDestroy(start));
      CHECK_CUDA(cudaEventDestroy(stop));
      CHECK_CUDA(cudaFree(dA));
      CHECK_CUDA(cudaFree(dB));
      CHECK_CUDA(cudaFree(dC));

      if (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim) {
        pm = ComputeMetrics(C_cpu.data(), C_gpu_fp32.data(), C_size);
      }
    }

    const double gflops = (gpu_ms > 0.f) ? (2.0 * M * N * K / (gpu_ms * 1e6)) : 0.0;
    std::cout << "\n========== " << M << "x" << N << "x" << K
              << (aligned ? "" : " (SKIP unaligned)")
              << " | GPU: " << std::fixed << std::setprecision(3) << gpu_ms << " ms"
              << " | " << std::setprecision(1) << gflops << " GFLOPS ==========\n";

    if (aligned && M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim) {
      std::cout << std::left << std::setw(16) << "CuTe FP16"
                << std::right << std::setw(12) << std::fixed << std::setprecision(6) << pm.cos_sim
                << std::setw(14) << std::setprecision(8) << pm.max_abs_err << "\n";
    }

    ofs << i << ",gemm_cute_fp16," << M << "," << N << "," << K << "," << gpu_ms << "," << gflops
        << "," << pm.cos_sim << "," << pm.snr_db << "," << pm.max_rel_err << "," << pm.mean_abs_err
        << "," << pm.p99_abs_err << "," << pm.max_abs_err << "\n";
  }
  return 0;
}
