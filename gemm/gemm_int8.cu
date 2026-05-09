#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>

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

// ============================================================
// GEMM INT8 WMMA Tensor Core  kernel
// ============================================================
namespace gemm_int8 {
namespace wmma = nvcuda::wmma;

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kTileK = 32;

constexpr int kNumWarpsM = 2;
constexpr int kNumWarpsN = 4;
constexpr int kWarpSize = 32;
constexpr int kThreads = kNumWarpsM * kNumWarpsN * kWarpSize;
static_assert(kThreads == 256, "thread count");

constexpr int kWarpTilesM = 4;
constexpr int kWarpTilesN = 2;
constexpr int kWarpM = kWarpTilesM * 16;
constexpr int kWarpN = kWarpTilesN * 16;
static_assert(kNumWarpsM * kWarpM == kBlockM, "block M coverage");
static_assert(kNumWarpsN * kWarpN == kBlockN, "block N coverage");

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
static_assert(kTileK % kWmmaK == 0, "kTileK must be multiple of kWmmaK");

constexpr int kBlockThreadsX = 16;
constexpr int kBlockThreadsY = 16;

constexpr int kSmemABuf = kBlockM * kTileK;
constexpr int kSmemBBuf = kTileK * kBlockN;
constexpr size_t kSmemSize = sizeof(int8_t) * 2 * (kSmemABuf + kSmemBBuf);

__global__ __launch_bounds__(kThreads, 2) void GemmINT8Kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ C,
    int M, int N, int K) {
  extern __shared__ int8_t shared_mem[];
  int8_t* As_buf0 = shared_mem;
  int8_t* As_buf1 = shared_mem + kSmemABuf;
  int8_t* Bs_buf0 = shared_mem + 2 * kSmemABuf;
  int8_t* Bs_buf1 = shared_mem + 2 * kSmemABuf + kSmemBBuf;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * kBlockThreadsX + tx;
  const int warp_id = tid / kWarpSize;
  const int warp_m = warp_id / kNumWarpsN;
  const int warp_n = warp_id % kNumWarpsN;

  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, int>
      c_frag[kWarpTilesM][kWarpTilesN];
  #pragma unroll
  for (int i = 0; i < kWarpTilesM; ++i) {
    #pragma unroll
    for (int j = 0; j < kWarpTilesN; ++j) {
      wmma::fill_fragment(c_frag[i][j], 0);
    }
  }

  const int num_k_tiles = (K + kTileK - 1) / kTileK;
  const int a_offset_base = warp_m * kWarpM;
  const int b_offset_base = warp_n * kWarpN;
  const int a_ld = kTileK;

  auto load_tile_async = [&](int tile_k_start, int8_t* As_buf, int8_t* Bs_buf) {
    const int total_a_int16 = (kBlockM * kTileK) / 16;
    const int a_int16_per_thread = (total_a_int16 + kThreads - 1) / kThreads;
    for (int l = 0; l < a_int16_per_thread; ++l) {
      int idx = tid * a_int16_per_thread + l;
      if (idx >= total_a_int16) continue;
      int r = idx / (kTileK / 16);
      int k_offset = (idx % (kTileK / 16)) * 16;
      int g_r = blockIdx.y * kBlockM + r;
      int g_k = tile_k_start + k_offset;
      int8_t* dst = As_buf + r * kTileK + k_offset;
      if (g_r < M && g_k + 15 < K) {
        __pipeline_memcpy_async(dst, &A[static_cast<size_t>(g_r) * K + g_k], 16);
      } else {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
          dst[i] = (g_r < M && g_k + i < K) ? A[static_cast<size_t>(g_r) * K + g_k + i] : int8_t(0);
        }
      }
    }

    const int total_b_int16 = (kTileK * kBlockN) / 16;
    const int b_int16_per_thread = (total_b_int16 + kThreads - 1) / kThreads;
    for (int l = 0; l < b_int16_per_thread; ++l) {
      int idx = tid * b_int16_per_thread + l;
      if (idx >= total_b_int16) continue;
      int k_idx = idx / (kBlockN / 16);
      int c_offset = (idx % (kBlockN / 16)) * 16;
      int g_k = tile_k_start + k_idx;
      int g_c = blockIdx.x * kBlockN + c_offset;
      int8_t* dst = Bs_buf + k_idx * kBlockN + c_offset;
      if (g_k < K && g_c + 15 < N) {
        __pipeline_memcpy_async(dst, &B[static_cast<size_t>(g_k) * N + g_c], 16);
      } else {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
          dst[i] = (g_k < K && g_c + i < N) ? B[static_cast<size_t>(g_k) * N + g_c + i] : int8_t(0);
        }
      }
    }
  };

  load_tile_async(0, As_buf0, Bs_buf0);
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();

  #pragma unroll
  for (int t = 0; t < num_k_tiles; ++t) {
    int8_t* As_read = (t & 1) ? As_buf1 : As_buf0;
    int8_t* Bs_read = (t & 1) ? Bs_buf1 : Bs_buf0;

    if (t + 1 < num_k_tiles) {
      int8_t* As_write = (t & 1) ? As_buf0 : As_buf1;
      int8_t* Bs_write = (t & 1) ? Bs_buf0 : Bs_buf1;
      load_tile_async((t + 1) * kTileK, As_write, Bs_write);
      __pipeline_commit();
    }

    #pragma unroll
    for (int kk = 0; kk < kTileK; kk += kWmmaK) {
      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, signed char, wmma::row_major>
          a_frag[kWarpTilesM];
      #pragma unroll
      for (int i = 0; i < kWarpTilesM; ++i) {
        wmma::load_matrix_sync(a_frag[i],
            (const signed char*)(As_read + (a_offset_base + i * kWmmaM) * a_ld + kk), a_ld);
      }

      #pragma unroll
      for (int j = 0; j < kWarpTilesN; ++j) {
        wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, signed char, wmma::row_major>
            b_frag;
        wmma::load_matrix_sync(b_frag,
            (const signed char*)(Bs_read + kk * kBlockN + b_offset_base + j * kWmmaN), kBlockN);

        #pragma unroll
        for (int i = 0; i < kWarpTilesM; ++i) {
          wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag, c_frag[i][j], false);
        }
      }
    }

    if (t + 1 < num_k_tiles) {
      __pipeline_wait_prior(0);
    }
    __syncthreads();
  }

  const int out_r = blockIdx.y * kBlockM + a_offset_base;
  const int out_c = blockIdx.x * kBlockN + b_offset_base;

  #pragma unroll
  for (int i = 0; i < kWarpTilesM; ++i) {
    #pragma unroll
    for (int j = 0; j < kWarpTilesN; ++j) {
      const int g_r = out_r + i * kWmmaM;
      const int g_c = out_c + j * kWmmaN;
      if (g_r + kWmmaM <= M && g_c + kWmmaN <= N) {
        wmma::store_matrix_sync(
            C + static_cast<size_t>(g_r) * N + g_c,
            c_frag[i][j], N, wmma::mem_row_major);
      }
    }
  }
}

}  // namespace gemm_int8

// ============================================================
// 精度评估指标
// ============================================================
struct PrecisionMetrics {
  double cos_sim;       // 余弦相似度
  double snr_db;        // 信噪比 (dB)
  double max_rel_err;   // 最大相对误差
  double mean_abs_err;  // 平均绝对误差
  double p99_abs_err;   // 99% 分位绝对误差
  double max_abs_err;   // 最大绝对误差
};

static PrecisionMetrics ComputeMetrics(const float* ref, const float* test, size_t n) {
  PrecisionMetrics m = {};
  double dot = 0.0, norm_ref = 0.0, norm_test = 0.0;
  double noise_power = 0.0, signal_power = 0.0;
  double max_rel = 0.0;
  std::vector<double> abs_errs(n);
  double sum_abs = 0.0;
  double max_abs = 0.0;

  // 先扫描 ref 的 max_abs，用于相对误差阈值
  double ref_max_abs = 0.0;
  for (size_t i = 0; i < n; ++i)
    ref_max_abs = std::max(ref_max_abs, static_cast<double>(std::fabs(ref[i])));
  double rel_threshold = std::max(ref_max_abs * 1e-6, 1e-6);

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
// 量化方案
// ============================================================

// Per-tensor 量化：整个矩阵共享一个 scale
static double QuantizePerTensor(const std::vector<float>& src, std::vector<int8_t>& dst, float& scale) {
  float max_abs = 0.0f;
  for (float v : src) max_abs = std::max(max_abs, std::fabs(v));
  scale = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
  double max_qerr = 0.0;
  for (size_t i = 0; i < src.size(); ++i) {
    float q = std::round(src[i] / scale);
    q = std::max(-128.0f, std::min(127.0f, q));
    dst[i] = static_cast<int8_t>(q);
    float err = std::fabs(src[i] - static_cast<float>(q) * scale);
    max_qerr = std::max(max_qerr, static_cast<double>(err));
  }
  return max_qerr;
}

// Per-channel 量化：权重 B (K×N) 每列（输出通道）独立 scale
// 激活 A (M×K) 仍用 per-tensor
static std::vector<float> QuantizePerChannelWeight(
    const std::vector<float>& src, std::vector<int8_t>& dst, int rows, int cols) {
  std::vector<float> scales(cols);
  for (int c = 0; c < cols; ++c) {
    float max_abs = 0.0f;
    for (int r = 0; r < rows; ++r) {
      max_abs = std::max(max_abs, std::fabs(src[r * cols + c]));
    }
    scales[c] = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
    for (int r = 0; r < rows; ++r) {
      size_t idx = r * cols + c;
      float q = std::round(src[idx] / scales[c]);
      q = std::max(-128.0f, std::min(127.0f, q));
      dst[idx] = static_cast<int8_t>(q);
    }
  }
  return scales;
}

// ============================================================
// 数据生成
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

// ============================================================
// CPU GEMM 参考实现
// ============================================================
static void GemmINT8_CPU(const std::vector<int8_t>& A, const std::vector<int8_t>& B,
                          std::vector<int32_t>& C, int M, int N, int K) {
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < N; ++c) {
      int32_t s = 0;
      for (int k = 0; k < K; ++k)
        s += static_cast<int32_t>(A[r * K + k]) * static_cast<int32_t>(B[k * N + c]);
      C[r * N + c] = s;
    }
}

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
// 方案评估
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

  if (scheme_name == "Per-Tensor INT8") {
    float scale_a, scale_b;
    std::vector<int8_t> A_int8(A_fp32.size()), B_int8(B_fp32.size());
    double qerr_a = QuantizePerTensor(A_fp32, A_int8, scale_a);
    double qerr_b = QuantizePerTensor(B_fp32, B_int8, scale_b);
    res.quant_err = std::max(qerr_a, qerr_b);

    std::vector<int32_t> C_int32(C_ref.size());
    GemmINT8_CPU(A_int8, B_int8, C_int32, M, N, K);

    std::vector<float> C_deq(C_ref.size());
    float scale_c = scale_a * scale_b;
    for (size_t i = 0; i < C_deq.size(); ++i)
      C_deq[i] = static_cast<float>(C_int32[i]) * scale_c;

    res.metrics = ComputeMetrics(C_ref.data(), C_deq.data(), C_ref.size());
    results.push_back(res);
    return;
  }

  if (scheme_name == "Per-Channel INT8") {
    // A per-tensor, B per-channel
    float scale_a;
    std::vector<int8_t> A_int8(A_fp32.size()), B_int8(B_fp32.size());
    QuantizePerTensor(A_fp32, A_int8, scale_a);
    std::vector<float> scales_b = QuantizePerChannelWeight(B_fp32, B_int8, K, N);

    // 量化误差：B 每列的最大误差
    double max_qerr = 0.0;
    for (int c = 0; c < N; ++c) {
      for (int r = 0; r < K; ++r) {
        size_t idx = r * N + c;
        float err = std::fabs(B_fp32[idx] - static_cast<float>(B_int8[idx]) * scales_b[c]);
        max_qerr = std::max(max_qerr, static_cast<double>(err));
      }
    }
    {
      float s;
      std::vector<int8_t> tmp(A_fp32.size());
      double qa = QuantizePerTensor(A_fp32, tmp, s);
      max_qerr = std::max(max_qerr, qa);
    }
    res.quant_err = max_qerr;

    std::vector<int32_t> C_int32(C_ref.size());
    GemmINT8_CPU(A_int8, B_int8, C_int32, M, N, K);

    // Per-channel 反量化：每列不同 scale
    std::vector<float> C_deq(C_ref.size());
    for (int r = 0; r < M; ++r)
      for (int c = 0; c < N; ++c)
        C_deq[r * N + c] = static_cast<float>(C_int32[r * N + c]) * scale_a * scales_b[c];

    res.metrics = ComputeMetrics(C_ref.data(), C_deq.data(), C_ref.size());
    results.push_back(res);
    return;
  }
}

// ============================================================
// 输出辅助
// ============================================================
static void PrintHeader() {
  std::cout << std::left << std::setw(22) << "Scheme"
            << std::right
            << std::setw(12) << "CosSim"
            << std::setw(12) << "SNR(dB)"
            << std::setw(14) << "MaxRelErr"
            << std::setw(14) << "MeanAbsErr"
            << std::setw(14) << "P99AbsErr"
            << std::setw(14) << "MaxAbsErr"
            << std::setw(12) << "QuantErr"
            << "\n";
  std::cout << std::string(22 + 12*7, '-') << "\n";
}

static void PrintResult(const SchemeResult& res) {
  std::cout << std::left << std::setw(22) << res.name
            << std::right
            << std::setw(12) << std::fixed << std::setprecision(6) << res.metrics.cos_sim
            << std::setw(12) << std::setprecision(2) << res.metrics.snr_db
            << std::setw(14) << std::setprecision(6) << res.metrics.max_rel_err
            << std::setw(14) << std::setprecision(8) << res.metrics.mean_abs_err
            << std::setw(14) << std::setprecision(8) << res.metrics.p99_abs_err
            << std::setw(14) << std::setprecision(8) << res.metrics.max_abs_err
            << std::setw(12) << std::setprecision(6) << res.quant_err
            << "\n";
}

static std::string MetricsToCSV(const PrecisionMetrics& m) {
  std::ostringstream oss;
  oss << std::setprecision(10)
      << m.cos_sim << ","
      << m.snr_db << ","
      << m.max_rel_err << ","
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
  std::ofstream ofs("data/results/gemm_int8_results.csv");
  ofs << "id,group,M,N,K,"
      << "gpu_ms,gflops,"
      << "pt_cos_sim,pt_snr_db,pt_max_rel_err,pt_mean_abs_err,pt_p99_abs_err,pt_max_abs_err,"
      << "pc_cos_sim,pc_snr_db,pc_max_rel_err,pc_mean_abs_err,pc_p99_abs_err,pc_max_abs_err\n";

  CHECK_CUDA(cudaFuncSetAttribute(gemm_int8::GemmINT8Kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      gemm_int8::kSmemSize));

  for (size_t i = 0; i < cases.size(); ++i) {
    const int M = cases[i].rows;
    const int N = cases[i].cols;
    const int K = M;
    const bool aligned = (M % gemm_int8::kBlockM == 0) &&
                         (N % gemm_int8::kBlockN == 0) &&
                         (K % gemm_int8::kTileK == 0);

    std::vector<float> A_fp32(static_cast<size_t>(M) * K);
    std::vector<float> B_fp32(static_cast<size_t>(K) * N);
    GenerateMatrices(A_fp32, B_fp32, M, K, N, static_cast<unsigned>(i + 42));

    const bool do_precision = (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim);
    const size_t C_size = static_cast<size_t>(M) * N;
    std::vector<float> C_ref;
    if (do_precision) {
      C_ref.resize(C_size);
      GemmFP32_CPU(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);
    }

    // GPU 性能（用 per-tensor 量化数据跑 kernel）
    float gpu_ms = 0.0f;
    double gflops = 0.0;

    if (aligned) {
      float scale_a_unused, scale_b_unused;
      std::vector<int8_t> A_int8(A_fp32.size()), B_int8(B_fp32.size());
      QuantizePerTensor(A_fp32, A_int8, scale_a_unused);
      QuantizePerTensor(B_fp32, B_int8, scale_b_unused);

      int8_t *dA, *dB;
      int32_t *dC;
      CHECK_CUDA(cudaMalloc(&dA, A_int8.size() * sizeof(int8_t)));
      CHECK_CUDA(cudaMalloc(&dB, B_int8.size() * sizeof(int8_t)));
      CHECK_CUDA(cudaMalloc(&dC, C_size * sizeof(int32_t)));
      CHECK_CUDA(cudaMemcpy(dA, A_int8.data(), A_int8.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(dB, B_int8.data(), B_int8.size() * sizeof(int8_t), cudaMemcpyHostToDevice));

      dim3 block(gemm_int8::kBlockThreadsX, gemm_int8::kBlockThreadsY);
      dim3 grid((N + gemm_int8::kBlockN - 1) / gemm_int8::kBlockN,
                (M + gemm_int8::kBlockM - 1) / gemm_int8::kBlockM);

      // warmup
      gemm_int8::GemmINT8Kernel<<<grid, block, gemm_int8::kSmemSize>>>(dA, dB, dC, M, N, K);
      CHECK_CUDA(cudaDeviceSynchronize());

      cudaEvent_t start, stop;
      CHECK_CUDA(cudaEventCreate(&start));
      CHECK_CUDA(cudaEventCreate(&stop));
      CHECK_CUDA(cudaEventRecord(start));
      for (int rep = 0; rep < kRepeat; ++rep) {
        gemm_int8::GemmINT8Kernel<<<grid, block, gemm_int8::kSmemSize>>>(dA, dB, dC, M, N, K);
      }
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));
      gpu_ms /= static_cast<float>(kRepeat);
      gflops = 2.0 * M * N * K / (gpu_ms * 1e6);

      CHECK_CUDA(cudaEventDestroy(start));
      CHECK_CUDA(cudaEventDestroy(stop));
      CHECK_CUDA(cudaFree(dA));
      CHECK_CUDA(cudaFree(dB));
      CHECK_CUDA(cudaFree(dC));
    }

    // 精度评估：FP32 vs Per-Tensor vs Per-Channel
    std::vector<SchemeResult> results;
    if (do_precision) {
      EvaluateScheme(A_fp32, B_fp32, M, N, K, C_ref, "FP32 Ref", results);
      EvaluateScheme(A_fp32, B_fp32, M, N, K, C_ref, "Per-Tensor INT8", results);
      EvaluateScheme(A_fp32, B_fp32, M, N, K, C_ref, "Per-Channel INT8", results);
    }

    // 输出
    std::cout << "\n========== " << M << "x" << N << "x" << K
              << " | GPU: " << std::fixed << std::setprecision(3) << gpu_ms << " ms"
              << " | " << std::setprecision(1) << gflops << " GFLOPS"
              << " | " << (aligned ? "ALIGNED" : "UNALIGNED") << " ==========\n";

    if (do_precision) {
      PrintHeader();
      for (const auto& r : results)
        PrintResult(r);
    } else {
      std::cout << "  (精度评估跳过：矩阵太大，仅上报 GPU 性能)\n";
    }

    // CSV 输出
    ofs << i << ",gemm_int8," << M << "," << N << "," << K << ","
        << gpu_ms << "," << gflops << ",";

    if (do_precision && results.size() >= 3) {
      ofs << MetricsToCSV(results[1].metrics) << ","
          << MetricsToCSV(results[2].metrics);
    } else {
      ofs << "0,0,0,0,0,0,"
          << "0,0,0,0,0,0";
    }
    ofs << "\n";
    ofs.flush();
  }

  std::cout << "\n结果已写入 data/results/gemm_int8_results.csv\n";

  return 0;
}
