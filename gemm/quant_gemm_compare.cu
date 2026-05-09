// Host-side numerical comparison: FP16 / FP8 E4M3 / INT8(per-tensor) round-trip
// and simulated GEMM (FP32 accumulation on dequantized operands).
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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

static double MaxRoundtripFp16(const std::vector<float>& src) {
  double m = 0.0;
  for (float v : src) {
    __half h = __float2half(v);
    float back = __half2float(h);
    m = std::max(m, static_cast<double>(std::fabs(v - back)));
  }
  return m;
}

static double MaxRoundtripFp8E4m3(const std::vector<float>& src) {
  double m = 0.0;
  for (float v : src) {
    __nv_fp8_e4m3 q(v);
    float back = static_cast<float>(q);
    m = std::max(m, static_cast<double>(std::fabs(v - back)));
  }
  return m;
}

static double QuantizeDequantMatrix(const std::vector<float>& src, std::vector<float>& dst,
                                    float& scale) {
  float max_abs = 0.0f;
  for (float v : src) max_abs = std::max(max_abs, std::fabs(v));
  scale = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
  dst.resize(src.size());
  double max_qerr = 0.0;
  for (size_t i = 0; i < src.size(); ++i) {
    float q = std::round(src[i] / scale);
    q = std::max(-128.0f, std::min(127.0f, q));
    dst[i] = q * scale;
    max_qerr = std::max(max_qerr, static_cast<double>(std::fabs(src[i] - dst[i])));
  }
  return max_qerr;
}

static void ApplyFp16Roundtrip(const std::vector<float>& src, std::vector<float>& dst) {
  dst.resize(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    __half h = __float2half(src[i]);
    dst[i] = __half2float(h);
  }
}

static void ApplyFp8Roundtrip(const std::vector<float>& src, std::vector<float>& dst) {
  dst.resize(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    __nv_fp8_e4m3 q(src[i]);
    dst[i] = static_cast<float>(q);
  }
}

static double MaxAbsDiffOut(const std::vector<float>& ref, const std::vector<float>& test) {
  double m = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    m = std::max(m, static_cast<double>(std::fabs(ref[i] - test[i])));
  }
  return m;
}

int main() {
  constexpr int kMaxCpuGemmDim = 1024;
  auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/quant_gemm_numerical.csv");
  ofs << "id,M,N,K,max_rt_fp16,max_rt_fp8_e4m3,max_rt_int8_dequant,gemm_err_fp16_inputs,"
         "gemm_err_fp8_inputs,gemm_err_int8_inputs\n";

  for (size_t i = 0; i < cases.size(); ++i) {
    const int M = cases[i].rows;
    const int N = cases[i].cols;
    const int K = M;

    std::vector<float> A(M * K), B(K * N);
    common::InitMatrix(A, M, K);
    common::InitMatrix(B, K, N);

    const double rt_fp16 = std::max(MaxRoundtripFp16(A), MaxRoundtripFp16(B));
    const double rt_fp8 = std::max(MaxRoundtripFp8E4m3(A), MaxRoundtripFp8E4m3(B));

    std::vector<float> Aq, Bq;
    float sa = 1.0f, sb = 1.0f;
    const double qe_a = QuantizeDequantMatrix(A, Aq, sa);
    const double qe_b = QuantizeDequantMatrix(B, Bq, sb);
    const double rt_int8 = std::max(qe_a, qe_b);

    double ge_fp16 = -1.0, ge_fp8 = -1.0, ge_int8 = -1.0;
    if (M <= kMaxCpuGemmDim && N <= kMaxCpuGemmDim && K <= kMaxCpuGemmDim) {
      std::vector<float> C_ref(static_cast<size_t>(M) * N);
      GemmCpuFp32(A.data(), B.data(), C_ref.data(), M, N, K);

      std::vector<float> A16, B16, C_t(static_cast<size_t>(M) * N);
      ApplyFp16Roundtrip(A, A16);
      ApplyFp16Roundtrip(B, B16);
      GemmCpuFp32(A16.data(), B16.data(), C_t.data(), M, N, K);
      ge_fp16 = MaxAbsDiffOut(C_ref, C_t);

      std::vector<float> A8, B8;
      ApplyFp8Roundtrip(A, A8);
      ApplyFp8Roundtrip(B, B8);
      GemmCpuFp32(A8.data(), B8.data(), C_t.data(), M, N, K);
      ge_fp8 = MaxAbsDiffOut(C_ref, C_t);

      GemmCpuFp32(Aq.data(), Bq.data(), C_t.data(), M, N, K);
      ge_int8 = MaxAbsDiffOut(C_ref, C_t);
    }

    auto fmt_gemm = [](double v) -> std::string {
      if (v < 0.0) return "SKIP_CPU_GEMM";
      std::ostringstream os;
      os << std::scientific << std::setprecision(6) << v;
      return os.str();
    };

    std::cout << M << "³"
              << " | rt_fp16=" << std::scientific << std::setprecision(3) << rt_fp16
              << " rt_fp8=" << rt_fp8 << " rt_int8=" << rt_int8 << std::fixed
              << " | gemm_out_err fp16=" << fmt_gemm(ge_fp16) << " fp8=" << fmt_gemm(ge_fp8)
              << " int8=" << fmt_gemm(ge_int8) << "\n";

    ofs << i << "," << M << "," << N << "," << K << "," << std::scientific << std::setprecision(6)
        << rt_fp16 << "," << rt_fp8 << "," << rt_int8 << ",";
    if (ge_fp16 >= 0.0)
      ofs << ge_fp16 << "," << ge_fp8 << "," << ge_int8 << "\n";
    else
      ofs << "SKIP,SKIP,SKIP\n";
  }
  return 0;
}
