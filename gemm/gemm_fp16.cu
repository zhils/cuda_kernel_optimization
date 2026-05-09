#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

namespace gemm_fp16 {
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
constexpr size_t kSmemSize = sizeof(__half) * 2 * (kSmemABuf + kSmemBBuf);

__global__ __launch_bounds__(kThreads, 2) void GemmFP16Kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
  extern __shared__ __half shared_mem[];
  __half* As_buf0 = shared_mem;
  __half* As_buf1 = shared_mem + kSmemABuf;
  __half* Bs_buf0 = shared_mem + 2 * kSmemABuf;
  __half* Bs_buf1 = shared_mem + 2 * kSmemABuf + kSmemBBuf;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * kBlockThreadsX + tx;
  const int warp_id = tid / kWarpSize;
  const int warp_m = warp_id / kNumWarpsN;
  const int warp_n = warp_id % kNumWarpsN;

  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>
      c_frag[kWarpTilesM][kWarpTilesN];
  #pragma unroll
  for (int i = 0; i < kWarpTilesM; ++i) {
    #pragma unroll
    for (int j = 0; j < kWarpTilesN; ++j) {
      wmma::fill_fragment(c_frag[i][j], 0.0f);
    }
  }

  const int num_k_tiles = (K + kTileK - 1) / kTileK;
  const int a_offset_base = warp_m * kWarpM;
  const int b_offset_base = warp_n * kWarpN;
  const int a_ld = kTileK;

  auto load_tile_async = [&](int tile_k_start, __half* As_buf, __half* Bs_buf) {
    const int total_a_half8 = (kBlockM * kTileK) / 8;
    const int a_half8_per_thread = (total_a_half8 + kThreads - 1) / kThreads;
    for (int l = 0; l < a_half8_per_thread; ++l) {
      int idx = tid * a_half8_per_thread + l;
      if (idx >= total_a_half8) continue;
      int r = idx / (kTileK / 8);
      int k_offset = (idx % (kTileK / 8)) * 8;
      int g_r = blockIdx.y * kBlockM + r;
      int g_k = tile_k_start + k_offset;
      __half* dst = As_buf + r * kTileK + k_offset;
      if (g_r < M && g_k + 7 < K) {
        __pipeline_memcpy_async(dst, &A[static_cast<size_t>(g_r) * K + g_k], 16);
      } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
          dst[i] = (g_r < M && g_k + i < K) ? A[static_cast<size_t>(g_r) * K + g_k + i] : __half(0.0f);
        }
      }
    }

    const int total_b_half8 = (kTileK * kBlockN) / 8;
    const int b_half8_per_thread = (total_b_half8 + kThreads - 1) / kThreads;
    for (int l = 0; l < b_half8_per_thread; ++l) {
      int idx = tid * b_half8_per_thread + l;
      if (idx >= total_b_half8) continue;
      int k_idx = idx / (kBlockN / 8);
      int c_offset = (idx % (kBlockN / 8)) * 8;
      int g_k = tile_k_start + k_idx;
      int g_c = blockIdx.x * kBlockN + c_offset;
      __half* dst = Bs_buf + k_idx * kBlockN + c_offset;
      if (g_k < K && g_c + 7 < N) {
        __pipeline_memcpy_async(dst, &B[static_cast<size_t>(g_k) * N + g_c], 16);
      } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
          dst[i] = (g_k < K && g_c + i < N) ? B[static_cast<size_t>(g_k) * N + g_c + i] : __half(0.0f);
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
    __half* As_read = (t & 1) ? As_buf1 : As_buf0;
    __half* Bs_read = (t & 1) ? Bs_buf1 : Bs_buf0;

    if (t + 1 < num_k_tiles) {
      __half* As_write = (t & 1) ? As_buf0 : As_buf1;
      __half* Bs_write = (t & 1) ? Bs_buf0 : Bs_buf1;
      load_tile_async((t + 1) * kTileK, As_write, Bs_write);
      __pipeline_commit();
    }

    #pragma unroll
    for (int kk = 0; kk < kTileK; kk += kWmmaK) {
      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __half, wmma::row_major>
          a_frag[kWarpTilesM];
      #pragma unroll
      for (int i = 0; i < kWarpTilesM; ++i) {
        wmma::load_matrix_sync(a_frag[i],
            As_read + (a_offset_base + i * kWmmaM) * a_ld + kk, a_ld);
      }

      #pragma unroll
      for (int j = 0; j < kWarpTilesN; ++j) {
        wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __half, wmma::row_major>
            b_frag;
        wmma::load_matrix_sync(b_frag,
            Bs_read + kk * kBlockN + b_offset_base + j * kWmmaN, kBlockN);

        #pragma unroll
        for (int i = 0; i < kWarpTilesM; ++i) {
          wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag, c_frag[i][j]);
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

}  // namespace gemm_fp16

// ============================================================
// 精度评估指标（与 gemm_int8 一致）
// ============================================================
struct PrecisionMetrics {
  double cos_sim;
  double snr_db;
  double max_rel_err;
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

static void GemmCPU_FP32(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c) {
      float s = 0.0f;
      for (int k = 0; k < K; ++k) s += A[static_cast<size_t>(r) * K + k] * B[static_cast<size_t>(k) * N + c];
      C[static_cast<size_t>(r) * N + c] = s;
    }
  }
}

int main() {
  constexpr int kRepeat = 10;
  constexpr int kMaxCpuVerifyDim = 1024;
  auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/gemm_fp16_results.csv");
  ofs << "id,group,M,N,K,gpu_ms,gflops,"
      << "cos_sim,snr_db,max_rel_err,mean_abs_err,p99_abs_err,max_abs_err\n";

  CHECK_CUDA(cudaFuncSetAttribute(gemm_fp16::GemmFP16Kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      gemm_fp16::kSmemSize));

  for (size_t i = 0; i < cases.size(); ++i) {
    const int M = cases[i].rows;
    const int N = cases[i].cols;
    const int K = M;
    const bool aligned = (M % gemm_fp16::kBlockM == 0) &&
                         (N % gemm_fp16::kBlockN == 0) &&
                         (K % gemm_fp16::kTileK == 0);
    const size_t C_size = static_cast<size_t>(M) * N;
    std::vector<float> A_fp32(static_cast<size_t>(M) * K),
                       B_fp32(static_cast<size_t>(K) * N),
                       C_cpu(C_size),
                       C_gpu_fp32(C_size);
    common::InitMatrix(A_fp32, M, K);
    common::InitMatrix(B_fp32, K, N);

    // FP32 CPU 参考
    if (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim) {
      GemmCPU_FP32(A_fp32.data(), B_fp32.data(), C_cpu.data(), M, N, K);
    }

    float gpu_ms = 0.0f;
    if (aligned) {
      std::vector<__half> A_half(A_fp32.size()), B_half(B_fp32.size());
      for (size_t j = 0; j < A_fp32.size(); ++j) A_half[j] = __float2half(A_fp32[j]);
      for (size_t j = 0; j < B_fp32.size(); ++j) B_half[j] = __float2half(B_fp32[j]);

      __half *dA, *dB;
      float *dC;
      CHECK_CUDA(cudaMalloc(&dA, A_half.size() * sizeof(__half)));
      CHECK_CUDA(cudaMalloc(&dB, B_half.size() * sizeof(__half)));
      CHECK_CUDA(cudaMalloc(&dC, C_size * sizeof(float)));
      CHECK_CUDA(cudaMemcpy(dA, A_half.data(), A_half.size() * sizeof(__half), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(dB, B_half.data(), B_half.size() * sizeof(__half), cudaMemcpyHostToDevice));

      dim3 block(gemm_fp16::kBlockThreadsX, gemm_fp16::kBlockThreadsY);
      dim3 grid((N + gemm_fp16::kBlockN - 1) / gemm_fp16::kBlockN,
                (M + gemm_fp16::kBlockM - 1) / gemm_fp16::kBlockM);

      gemm_fp16::GemmFP16Kernel<<<grid, block, gemm_fp16::kSmemSize>>>(dA, dB, dC, M, N, K);
      CHECK_CUDA(cudaDeviceSynchronize());

      CHECK_CUDA(cudaMemcpy(C_gpu_fp32.data(), dC, C_size * sizeof(float), cudaMemcpyDeviceToHost));

      cudaEvent_t start, stop;
      CHECK_CUDA(cudaEventCreate(&start));
      CHECK_CUDA(cudaEventCreate(&stop));
      CHECK_CUDA(cudaEventRecord(start));
      for (int rep = 0; rep < kRepeat; ++rep) {
        gemm_fp16::GemmFP16Kernel<<<grid, block, gemm_fp16::kSmemSize>>>(dA, dB, dC, M, N, K);
      }
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));
      gpu_ms /= static_cast<float>(kRepeat);

      CHECK_CUDA(cudaEventDestroy(start));
      CHECK_CUDA(cudaEventDestroy(stop));
      CHECK_CUDA(cudaFree(dA));
      CHECK_CUDA(cudaFree(dB));
      CHECK_CUDA(cudaFree(dC));
    }

    // 精度评估：FP32 参考 vs FP16 GPU 结果
    double max_abs_diff = 0.0;
    PrecisionMetrics pm = {};
    if (aligned && M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim) {
      pm = ComputeMetrics(C_cpu.data(), C_gpu_fp32.data(), C_size);
      max_abs_diff = pm.max_abs_err;
    }

    const double gflops = (gpu_ms > 0.0f) ? (2.0 * M * N * K / (gpu_ms * 1e6)) : 0.0;
    std::cout << "\n========== " << M << "x" << N << "x" << K
              << " | GPU: " << std::fixed << std::setprecision(3) << gpu_ms << " ms"
              << " | " << std::setprecision(1) << gflops << " GFLOPS ==========\n";

    if (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim) {
      std::cout << std::left << std::setw(16) << "Scheme"
                << std::right
                << std::setw(12) << "CosSim"
                << std::setw(12) << "SNR(dB)"
                << std::setw(14) << "MaxRelErr"
                << std::setw(14) << "MeanAbsErr"
                << std::setw(14) << "P99AbsErr"
                << std::setw(14) << "MaxAbsErr"
                << "\n";
      std::cout << std::string(16 + 12*6, '-') << "\n";
      std::cout << std::left << std::setw(16) << "FP16 WMMA"
                << std::right
                << std::setw(12) << std::fixed << std::setprecision(6) << pm.cos_sim
                << std::setw(12) << std::setprecision(2) << pm.snr_db
                << std::setw(14) << std::setprecision(6) << pm.max_rel_err
                << std::setw(14) << std::setprecision(8) << pm.mean_abs_err
                << std::setw(14) << std::setprecision(8) << pm.p99_abs_err
                << std::setw(14) << std::setprecision(8) << pm.max_abs_err
                << "\n";
    }

    ofs << i << ",gemm_fp16," << M << "," << N << "," << K << ","
        << gpu_ms << "," << gflops << ","
        << pm.cos_sim << "," << pm.snr_db << "," << pm.max_rel_err << ","
        << pm.mean_abs_err << "," << pm.p99_abs_err << "," << pm.max_abs_err << "\n";
  }
  return 0;
}
