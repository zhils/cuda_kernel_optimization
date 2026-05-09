// FP8 E4M3 + per-column bias correction: CPU simulation
// Measures how much bias correction can improve FP8 GEMM accuracy.
// Also compares against INT8 bias correction on the same GEMM setup.
// Reference: FP32 full-precision GEMM.
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "common/benchmark.h"

static void GemmCpuFp32(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c) {
      float s = 0.0f;
      for (int k = 0; k < K; ++k) {
        s += A[static_cast<size_t>(r) * K + k] * B[static_cast<size_t>(k) * N + c];
      }
      C[static_cast<size_t>(r) * N + c] = s;
    }
  }
}

static void ApplyFp8Roundtrip(const std::vector<float>& src, std::vector<float>& dst) {
  dst.resize(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    __nv_fp8_e4m3 q(src[i]);
    dst[i] = static_cast<float>(q);
  }
}

static void ApplyInt8Roundtrip(const std::vector<float>& src, std::vector<float>& dst,
                               float& scale) {
  float max_abs = 0.0f;
  for (float v : src) max_abs = std::max(max_abs, std::fabs(v));
  scale = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
  dst.resize(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    float q = std::round(src[i] / scale);
    q = std::max(-128.0f, std::min(127.0f, q));
    dst[i] = q * scale;
  }
}

static void ComputeBiasCorrection(const std::vector<float>& ref,
                                  const std::vector<float>& quant,
                                  std::vector<float>& corr,
                                  int M_calib, int N) {
  corr.assign(N, 0.0);
  for (int r = 0; r < M_calib; ++r) {
    for (int c = 0; c < N; ++c) {
      corr[c] += ref[static_cast<size_t>(r) * N + c] - quant[static_cast<size_t>(r) * N + c];
    }
  }
  double inv = 1.0 / M_calib;
  for (int c = 0; c < N; ++c) corr[c] = static_cast<float>(corr[c] * inv);
}

static void ApplyBiasCorrection(const std::vector<float>& quant,
                                const std::vector<float>& corr,
                                std::vector<float>& corrected,
                                int M, int N) {
  corrected.resize(quant.size());
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c) {
      corrected[static_cast<size_t>(r) * N + c] =
          quant[static_cast<size_t>(r) * N + c] + corr[c];
    }
  }
}

static double MaxAbsDiffOut(const std::vector<float>& ref, const std::vector<float>& test) {
  double m = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    m = std::max(m, static_cast<double>(std::fabs(ref[i] - test[i])));
  }
  return m;
}

static double MeanAbsErr(const std::vector<float>& ref, const std::vector<float>& test) {
  double s = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    s += std::fabs(static_cast<double>(ref[i] - test[i]));
  }
  return s / static_cast<double>(ref.size());
}

struct Result {
  double mae_before, mae_after, maxabs_before, maxabs_after;
  double mae_red_pct, maxabs_red_pct;
};

static Result RunTest(const std::vector<float>& A, const std::vector<float>& B,
                      const std::vector<float>& C_ref,
                      int M, int N, int K, const std::string& label,
                      void (*roundtrip_fn)(const std::vector<float>&, std::vector<float>&)) {
  int M_calib = std::max(1, M / 2);
  int M_test = M - M_calib;

  std::vector<float> A_q, B_q;
  roundtrip_fn(A, A_q);
  roundtrip_fn(B, B_q);
  std::vector<float> C_q(static_cast<size_t>(M) * N);
  GemmCpuFp32(A_q.data(), B_q.data(), C_q.data(), M, N, K);

  std::vector<float> corr;
  ComputeBiasCorrection(C_ref, C_q, corr, M_calib, N);

  std::vector<float> C_q_test(C_q.begin() + static_cast<size_t>(M_calib) * N, C_q.end());
  std::vector<float> C_ref_test(C_ref.begin() + static_cast<size_t>(M_calib) * N, C_ref.end());
  std::vector<float> C_corrected;
  ApplyBiasCorrection(C_q_test, corr, C_corrected, M_test, N);

  Result r;
  r.mae_before = MeanAbsErr(C_ref_test, C_q_test);
  r.maxabs_before = MaxAbsDiffOut(C_ref_test, C_q_test);
  r.mae_after = MeanAbsErr(C_ref_test, C_corrected);
  r.maxabs_after = MaxAbsDiffOut(C_ref_test, C_corrected);
  r.mae_red_pct = (r.mae_before > 0.0) ? (1.0 - r.mae_after / r.mae_before) * 100.0 : 0.0;
  r.maxabs_red_pct = (r.maxabs_before > 0.0) ? (1.0 - r.maxabs_after / r.maxabs_before) * 100.0 : 0.0;

  std::cout << "  " << label << " MAE:      " << std::scientific << std::setprecision(6) << r.mae_before
            << "  →  " << r.mae_after << "  (" << std::fixed << std::setprecision(2) << r.mae_red_pct << "%)\n";
  std::cout << "  " << label << " MaxAbsDiff: " << std::scientific << std::setprecision(6) << r.maxabs_before
            << "  →  " << r.maxabs_after << "  (" << std::fixed << std::setprecision(2) << r.maxabs_red_pct << "%)\n";

  return r;
}

// Wrapper that applies FP8 roundtrip
static void Fp8Roundtrip(const std::vector<float>& src, std::vector<float>& dst) {
  ApplyFp8Roundtrip(src, dst);
}

// Wrapper that applies INT8 per-tensor roundtrip
static void Int8Roundtrip(const std::vector<float>& src, std::vector<float>& dst) {
  float scale;
  ApplyInt8Roundtrip(src, dst, scale);
}

int main() {
  constexpr int kMaxCpuGemmDim = 1024;

  auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/quant_fp8_bias_compensate.csv");
  ofs << "id,M,N,K,"
      << "fp8_mae_before,fp8_mae_after,fp8_mae_red_pct,"
      << "fp8_maxabs_before,fp8_maxabs_after,fp8_maxabs_red_pct,"
      << "int8_mae_before,int8_mae_after,int8_mae_red_pct,"
      << "int8_maxabs_before,int8_maxabs_after,int8_maxabs_red_pct\n";

  std::cout << std::string(100, '=') << "\n";
  std::cout << "FP8 E4M3 vs INT8: Per-Column Bias Correction on GEMM (CPU Simulation)\n";
  std::cout << "Calibration: first 50% rows  |  Test: remaining 50% rows\n";
  std::cout << std::string(100, '=') << "\n\n";

  for (size_t i = 0; i < cases.size(); ++i) {
    const int M = cases[i].rows;
    const int N = cases[i].cols;
    const int K = M;

    if (M > kMaxCpuGemmDim || N > kMaxCpuGemmDim) {
      std::cout << M << "x" << N << " | SKIP (too large for CPU GEMM)\n\n";
      continue;
    }

    std::vector<float> A(M * K), B(K * N);
    common::InitMatrix(A, M, K);
    common::InitMatrix(B, K, N);

    std::vector<float> C_ref(static_cast<size_t>(M) * N);
    GemmCpuFp32(A.data(), B.data(), C_ref.data(), M, N, K);

    std::cout << M << "x" << N << " (calib=" << (M/2) << " rows, test=" << (M - M/2) << " rows)\n";

    Result fp8 = RunTest(A, B, C_ref, M, N, K, "FP8 E4M3", Fp8Roundtrip);
    Result int8 = RunTest(A, B, C_ref, M, N, K, "INT8    ", Int8Roundtrip);

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  >> Bias correction efficacy: FP8 MAE ↓" << fp8.mae_red_pct
              << "%  vs  INT8 MAE ↓" << int8.mae_red_pct << "%\n\n";

    ofs << i << "," << M << "," << N << "," << K << ","
        << std::scientific << std::setprecision(8)
        << fp8.mae_before << "," << fp8.mae_after << "," << std::fixed << std::setprecision(4)
        << fp8.mae_red_pct << ","
        << std::scientific << std::setprecision(8)
        << fp8.maxabs_before << "," << fp8.maxabs_after << "," << std::fixed << std::setprecision(4)
        << fp8.maxabs_red_pct << ","
        << std::scientific << std::setprecision(8)
        << int8.mae_before << "," << int8.mae_after << "," << std::fixed << std::setprecision(4)
        << int8.mae_red_pct << ","
        << std::scientific << std::setprecision(8)
        << int8.maxabs_before << "," << int8.maxabs_after << "," << std::fixed << std::setprecision(4)
        << int8.maxabs_red_pct << "\n";
  }
  return 0;
}
