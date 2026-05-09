#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

#define CHECK_CUBLAS(call)                                                       \
  do {                                                                           \
    cublasStatus_t s__ = (call);                                                 \
    if (s__ != CUBLAS_STATUS_SUCCESS) {                                          \
      std::cerr << "cuBLAS error: " << static_cast<int>(s__) << " at "           \
                << __FILE__ << ":" << __LINE__ << std::endl;                    \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

// ============================================================
// FP8 E4M3 Constants
// ============================================================
constexpr float kFP8E4M3_MAX = 448.0f;

// ============================================================
// Precision Metrics
// ============================================================
struct PrecisionMetrics {
  double cos_sim;
  double snr_db;
  double max_rel_err;
  double max_rel_by_range;
  double mean_abs_err;
  double p99_abs_err;
  double max_abs_err;
};

static PrecisionMetrics ComputeMetrics(const float* ref, const float* test, size_t n) {
  PrecisionMetrics m = {};
  double dot = 0.0, norm_ref = 0.0, norm_test = 0.0;
  double noise_power = 0.0, signal_power = 0.0;
  double max_rel = 0.0;
  std::vector<double> abs_errs(n);
  double sum_abs = 0.0;
  double max_abs = 0.0;

  double ref_max_abs = 0.0;
  double ref_min = ref[0], ref_max = ref[0];
  for (size_t i = 0; i < n; ++i) {
    ref_max_abs = std::max(ref_max_abs, static_cast<double>(std::fabs(ref[i])));
    ref_min = std::min(ref_min, static_cast<double>(ref[i]));
    ref_max = std::max(ref_max, static_cast<double>(ref[i]));
  }
  double rel_threshold = std::max(ref_max_abs * 1e-6, 1e-6);
  double ref_range = std::max(ref_max - ref_min, 1e-10);

  for (size_t i = 0; i < n; ++i) {
    double r = ref[i], t = test[i];
    dot += r * t;
    norm_ref += r * r;
    norm_test += t * t;
    double diff = r - t;
    noise_power += diff * diff;
    signal_power += r * r;
    double ab = std::fabs(diff);
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
  m.max_rel_by_range = max_abs / ref_range;
  m.mean_abs_err = sum_abs / n;
  m.max_abs_err = max_abs;

  if (n > 0) {
    size_t p99_idx = static_cast<size_t>(n * 0.99);
    p99_idx = std::min(p99_idx, n - 1);
    std::nth_element(abs_errs.begin(), abs_errs.begin() + p99_idx, abs_errs.end());
    m.p99_abs_err = abs_errs[p99_idx];
  }

  return m;
}

// ============================================================
// FP8 Quantization Functions
// ============================================================

static void QuantizeFP8PerTensor(const std::vector<float>& src,
                                  std::vector<__nv_fp8_e4m3>& dst,
                                  float& scale) {
  float max_abs = 0.0f;
  for (float v : src) max_abs = std::max(max_abs, std::fabs(v));
  scale = (max_abs > 0.0f) ? max_abs / kFP8E4M3_MAX : 1.0f;
  scale = std::max(scale, 1e-8f);
  dst.resize(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    float scaled = src[i] / scale;
    scaled = std::max(-kFP8E4M3_MAX, std::min(kFP8E4M3_MAX, scaled));
    dst[i] = __nv_fp8_e4m3(scaled);
  }
}

static std::vector<float> QuantizeFP8PerChannelWeight(
    const std::vector<float>& src,
    std::vector<__nv_fp8_e4m3>& dst,
    int rows, int cols) {
  std::vector<float> scales(cols);
  dst.resize(src.size());
  for (int c = 0; c < cols; ++c) {
    float max_abs = 0.0f;
    for (int r = 0; r < rows; ++r) {
      max_abs = std::max(max_abs, std::fabs(src[r * cols + c]));
    }
    float s = (max_abs > 0.0f) ? max_abs / kFP8E4M3_MAX : 1.0f;
    s = std::max(s, 1e-8f);
    scales[c] = s;
    for (int r = 0; r < rows; ++r) {
      size_t idx = r * cols + c;
      float scaled = src[idx] / s;
      scaled = std::max(-kFP8E4M3_MAX, std::min(kFP8E4M3_MAX, scaled));
      dst[idx] = __nv_fp8_e4m3(scaled);
    }
  }
  return scales;
}

// ============================================================
// Data Generation
// ============================================================
static void GenerateMatrices(std::vector<float>& A, std::vector<float>& B,
                              int M, int K, int N, unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int r = 0; r < M; ++r)
    for (int j = 0; j < K; ++j)
      A[r * K + j] = dist(rng);
  for (int j = 0; j < K; ++j)
    for (int c = 0; c < N; ++c)
      B[j * N + c] = dist(rng);
}

static void InjectActivationOutliers(std::vector<float>& A,
                                      int M, int K,
                                      std::mt19937& gen) {
  std::uniform_int_distribution<int> r_dist(0, M - 1);
  std::uniform_int_distribution<int> c_dist(0, K - 1);
  std::uniform_real_distribution<float> amp_dist(6.0f, 10.0f);
  int num_outliers = std::max(1, (M * K) / 64);
  for (int i = 0; i < num_outliers; ++i) {
    int r = r_dist(gen);
    int c = c_dist(gen);
    float amp = amp_dist(gen);
    A[r * K + c] += (amp * ((i & 1) ? 1.0f : -1.0f));
  }
}

// ============================================================
// CPU GEMM Reference
// ============================================================
static void GemmFP32_CPU(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < N; ++c) {
      float s = 0.0f;
      for (int k = 0; k < K; ++k)
        s += A[r * K + k] * B[k * N + c];
      C[r * N + c] = s;
    }
}

// ============================================================
// GPU cuBLASLt FP8 GEMM Performance
// ============================================================
static float RunCublasLtFP8Gemm(
    cublasLtHandle_t lt_handle,
    const __nv_fp8_e4m3* dA, const __nv_fp8_e4m3* dB, float* dC,
    int M, int N, int K,
    float scale_a, float scale_b,
    int num_repeats) {

  cublasLtMatmulDesc_t operation_desc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;

  cudaDataType_t fp8_type = CUDA_R_8F_E4M3;
  cudaDataType_t fp32_type = CUDA_R_32F;

  CHECK_CUBLAS(cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  float alpha = 1.0f, beta = 0.0f;
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operation_desc,
      CUBLASLT_MATMUL_DESC_SCALE_TYPE, &fp32_type, sizeof(fp32_type)));

  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, fp8_type, K, M, K));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, fp8_type, N, K, N));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, fp32_type, N, M, N));

  // Warmup
  CHECK_CUBLAS(cublasLtMatmul(lt_handle, operation_desc,
      &alpha, dA, Adesc, dB, Bdesc, &beta, dC, Cdesc, dC, Cdesc, nullptr, nullptr, 0, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int rep = 0; rep < num_repeats; ++rep) {
    CHECK_CUBLAS(cublasLtMatmul(lt_handle, operation_desc,
        &alpha, dA, Adesc, dB, Bdesc, &beta, dC, Cdesc, dC, Cdesc, nullptr, nullptr, 0, nullptr));
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  ms /= static_cast<float>(num_repeats);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Cdesc));
  CHECK_CUBLAS(cublasLtMatmulDescDestroy(operation_desc));

  return ms;
}

// ============================================================
// Scheme Evaluation (CPU-based accuracy)
// ============================================================
struct SchemeResult {
  std::string name;
  PrecisionMetrics metrics;
  double quant_err;
};

static void EvaluateScheme(
    const std::vector<float>& A_fp32,
    const std::vector<float>& B_fp32,
    int M, int N, int K,
    const std::vector<float>& C_ref,
    const std::string& scheme_name,
    std::vector<SchemeResult>& results) {

  SchemeResult res;
  res.name = scheme_name;

  if (scheme_name == "FP32 Ref") {
    PrecisionMetrics ideal = ComputeMetrics(C_ref.data(), C_ref.data(), C_ref.size());
    res.metrics = ideal;
    res.quant_err = 0.0;
    results.push_back(res);
    return;
  }

  if (scheme_name == "Per-Tensor FP8") {
    float scale_a, scale_b;
    std::vector<__nv_fp8_e4m3> A_fp8(A_fp32.size()), B_fp8(B_fp32.size());
    QuantizeFP8PerTensor(A_fp32, A_fp8, scale_a);
    QuantizeFP8PerTensor(B_fp32, B_fp8, scale_b);

    double max_qerr = 0.0;
    for (size_t i = 0; i < A_fp32.size(); ++i) {
      float deq = static_cast<float>(A_fp8[i]) * scale_a;
      max_qerr = std::max(max_qerr, static_cast<double>(std::fabs(A_fp32[i] - deq)));
    }
    for (size_t i = 0; i < B_fp32.size(); ++i) {
      float deq = static_cast<float>(B_fp8[i]) * scale_b;
      max_qerr = std::max(max_qerr, static_cast<double>(std::fabs(B_fp32[i] - deq)));
    }
    res.quant_err = max_qerr;

    std::vector<float> A_deq(A_fp32.size()), B_deq(B_fp32.size());
    for (size_t i = 0; i < A_deq.size(); ++i)
      A_deq[i] = static_cast<float>(A_fp8[i]) * scale_a;
    for (size_t i = 0; i < B_deq.size(); ++i)
      B_deq[i] = static_cast<float>(B_fp8[i]) * scale_b;

    std::vector<float> C_fp8(C_ref.size());
    GemmFP32_CPU(A_deq.data(), B_deq.data(), C_fp8.data(), M, N, K);

    res.metrics = ComputeMetrics(C_ref.data(), C_fp8.data(), C_ref.size());
    results.push_back(res);
    return;
  }

  if (scheme_name == "Per-Channel FP8") {
    float scale_a;
    std::vector<__nv_fp8_e4m3> A_fp8(A_fp32.size()), B_fp8(B_fp32.size());
    QuantizeFP8PerTensor(A_fp32, A_fp8, scale_a);
    std::vector<float> scales_b = QuantizeFP8PerChannelWeight(B_fp32, B_fp8, K, N);

    std::vector<float> A_deq(A_fp32.size());
    for (size_t i = 0; i < A_deq.size(); ++i)
      A_deq[i] = static_cast<float>(A_fp8[i]) * scale_a;

    std::vector<float> B_deq(B_fp32.size());
    for (int r = 0; r < K; ++r)
      for (int c = 0; c < N; ++c)
        B_deq[r * N + c] = static_cast<float>(B_fp8[r * N + c]) * scales_b[c];

    double max_qerr = 0.0;
    for (size_t i = 0; i < A_fp32.size(); ++i) {
      float deq = static_cast<float>(A_fp8[i]) * scale_a;
      max_qerr = std::max(max_qerr, static_cast<double>(std::fabs(A_fp32[i] - deq)));
    }
    for (size_t i = 0; i < B_fp32.size(); ++i) {
      float err = std::fabs(B_fp32[i] - B_deq[i]);
      max_qerr = std::max(max_qerr, static_cast<double>(err));
    }
    res.quant_err = max_qerr;

    std::vector<float> C_fp8(C_ref.size());
    GemmFP32_CPU(A_deq.data(), B_deq.data(), C_fp8.data(), M, N, K);

    res.metrics = ComputeMetrics(C_ref.data(), C_fp8.data(), C_ref.size());
    results.push_back(res);
    return;
  }
}

// ============================================================
// Output Helpers
// ============================================================
static void PrintHeader() {
  std::cout << std::left << std::setw(24) << "Scheme"
            << std::right
            << std::setw(12) << "CosSim"
            << std::setw(12) << "SNR(dB)"
            << std::setw(14) << "MaxRelErr"
            << std::setw(14) << "MaxRelRange"
            << std::setw(14) << "MeanAbsErr"
            << std::setw(14) << "P99AbsErr"
            << std::setw(14) << "MaxAbsErr"
            << "\n";
  std::cout << std::string(24 + 12 + 12 + 14*6, '-') << "\n";
}

static void PrintResult(const SchemeResult& res) {
  std::cout << std::left << std::setw(24) << res.name
            << std::right
            << std::setw(12) << std::fixed << std::setprecision(6) << res.metrics.cos_sim
            << std::setw(12) << std::setprecision(2) << res.metrics.snr_db
            << std::setw(14) << std::setprecision(6) << res.metrics.max_rel_err
            << std::setw(14) << std::setprecision(6) << res.metrics.max_rel_by_range
            << std::setw(14) << std::setprecision(8) << res.metrics.mean_abs_err
            << std::setw(14) << std::setprecision(8) << res.metrics.p99_abs_err
            << std::setw(14) << std::setprecision(8) << res.metrics.max_abs_err
            << "\n";
}

static std::string MetricsToCSV(const PrecisionMetrics& m) {
  std::ostringstream oss;
  oss << std::setprecision(10)
      << m.cos_sim << ","
      << m.snr_db << ","
      << m.max_rel_err << ","
      << m.max_rel_by_range << ","
      << m.mean_abs_err << ","
      << m.p99_abs_err << ","
      << m.max_abs_err;
  return oss.str();
}

// ============================================================
// main
// ============================================================
int main() {
  constexpr int kRepeat = 3;
  constexpr int kMaxCpuVerifyDim = 1024;
  auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/gemm_fp8_results.csv");
  ofs << "id,group,M,N,K,"
      << "gpu_ms,gflops,"
      << "pt_cos_sim,pt_snr_db,pt_max_rel_err,pt_max_rel_by_range,pt_mean_abs_err,pt_p99_abs_err,pt_max_abs_err,"
      << "pc_cos_sim,pc_snr_db,pc_max_rel_err,pc_max_rel_by_range,pc_mean_abs_err,pc_p99_abs_err,pc_max_abs_err\n";

  cublasLtHandle_t lt_handle;
  CHECK_CUBLAS(cublasLtCreate(&lt_handle));

  for (size_t i = 0; i < cases.size(); ++i) {
    const int M = cases[i].rows;
    const int N = cases[i].cols;
    const int K = M;

    std::vector<float> A_fp32(static_cast<size_t>(M) * K);
    std::vector<float> B_fp32(static_cast<size_t>(K) * N);
    GenerateMatrices(A_fp32, B_fp32, M, K, N, static_cast<unsigned>(i + 42));

    std::mt19937 outlier_gen(static_cast<unsigned>(i + 100));
    InjectActivationOutliers(A_fp32, M, K, outlier_gen);

    const bool do_precision = (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim);
    const size_t C_size = static_cast<size_t>(M) * N;
    std::vector<float> C_ref;
    if (do_precision) {
      C_ref.resize(C_size);
      GemmFP32_CPU(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);
    }

    // GPU Performance: cuBLASLt FP8 GEMM
    float gpu_ms = 0.0f;
    double gflops = 0.0;

    if (M % 16 == 0 && N % 16 == 0 && K % 16 == 0) {
      float scale_a, scale_b;
      std::vector<__nv_fp8_e4m3> A_fp8, B_fp8;
      QuantizeFP8PerTensor(A_fp32, A_fp8, scale_a);
      QuantizeFP8PerTensor(B_fp32, B_fp8, scale_b);

      __nv_fp8_e4m3 *dA, *dB;
      float *dC;
      CHECK_CUDA(cudaMalloc(&dA, A_fp8.size() * sizeof(__nv_fp8_e4m3)));
      CHECK_CUDA(cudaMalloc(&dB, B_fp8.size() * sizeof(__nv_fp8_e4m3)));
      CHECK_CUDA(cudaMalloc(&dC, C_size * sizeof(float)));
      CHECK_CUDA(cudaMemcpy(dA, A_fp8.data(), A_fp8.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(dB, B_fp8.data(), B_fp8.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));

      gpu_ms = RunCublasLtFP8Gemm(lt_handle, dA, dB, dC, M, N, K,
                                   scale_a, scale_b, kRepeat);
      gflops = 2.0 * M * N * K / (gpu_ms * 1e6);

      CHECK_CUDA(cudaFree(dA));
      CHECK_CUDA(cudaFree(dB));
      CHECK_CUDA(cudaFree(dC));
    }

    // Accuracy: FP32 vs Per-Tensor FP8 vs Per-Channel FP8
    std::vector<SchemeResult> results;
    if (do_precision) {
      EvaluateScheme(A_fp32, B_fp32, M, N, K, C_ref, "FP32 Ref", results);
      EvaluateScheme(A_fp32, B_fp32, M, N, K, C_ref, "Per-Tensor FP8", results);
      EvaluateScheme(A_fp32, B_fp32, M, N, K, C_ref, "Per-Channel FP8", results);
    }

    std::cout << "\n========== " << M << "x" << N << "x" << K
              << " | GPU FP8: " << std::fixed << std::setprecision(3) << gpu_ms << " ms"
              << " | " << std::setprecision(1) << gflops << " GFLOPS"
              << " | " << ((M % 16 == 0) ? "ALIGNED" : "UNALIGNED") << " ==========\n";

    if (do_precision) {
      PrintHeader();
      for (const auto& r : results)
        PrintResult(r);
    } else {
      std::cout << "  (Skipping accuracy eval: matrix too large for CPU reference)\n";
    }

    ofs << i << ",gemm_fp8," << M << "," << N << "," << K << ","
        << gpu_ms << "," << gflops << ",";

    if (do_precision && results.size() >= 3) {
      ofs << MetricsToCSV(results[1].metrics) << ","
          << MetricsToCSV(results[2].metrics);
    } else {
      ofs << "0,0,0,0,0,0,0,"
          << "0,0,0,0,0,0,0";
    }
    ofs << "\n";
    ofs.flush();
  }

  CHECK_CUBLAS(cublasLtDestroy(lt_handle));

  std::cout << "\nResults saved to data/results/gemm_fp8_results.csv\n";

  return 0;
}
