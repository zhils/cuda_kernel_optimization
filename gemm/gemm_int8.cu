#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

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

static double QuantizeMatrix(const std::vector<float>& src, std::vector<int8_t>& dst, float& scale) {
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
  constexpr int kRepeat = 3;
  constexpr int kMaxCpuVerifyDim = 1024;
  auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/gemm_int8_results.csv");
  ofs << "id,group,M,N,K,gpu_ms,gflops,max_abs_diff,check,quant_err\n";

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
    std::vector<float> A_fp32(static_cast<size_t>(M) * K),
                       B_fp32(static_cast<size_t>(K) * N),
                       C_cpu(static_cast<size_t>(M) * N),
                       C_gpu_fp32(static_cast<size_t>(M) * N);
    common::InitMatrix(A_fp32, M, K);
    common::InitMatrix(B_fp32, K, N);

    if (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim) {
      GemmCPU_FP32(A_fp32.data(), B_fp32.data(), C_cpu.data(), M, N, K);
    }

    float scale_a = 1.0f, scale_b = 1.0f;
    std::vector<int8_t> A_int8(A_fp32.size()), B_int8(B_fp32.size());
    double quant_err = QuantizeMatrix(A_fp32, A_int8, scale_a);
    quant_err = std::max(quant_err, QuantizeMatrix(B_fp32, B_int8, scale_b));
    const float scale_c = scale_a * scale_b;

    float gpu_ms = 0.0f;
    if (aligned) {
      int8_t *dA, *dB;
      int32_t *dC;
      CHECK_CUDA(cudaMalloc(&dA, A_int8.size() * sizeof(int8_t)));
      CHECK_CUDA(cudaMalloc(&dB, B_int8.size() * sizeof(int8_t)));
      CHECK_CUDA(cudaMalloc(&dC, C_gpu_fp32.size() * sizeof(int32_t)));
      CHECK_CUDA(cudaMemcpy(dA, A_int8.data(), A_int8.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(dB, B_int8.data(), B_int8.size() * sizeof(int8_t), cudaMemcpyHostToDevice));

      dim3 block(gemm_int8::kBlockThreadsX, gemm_int8::kBlockThreadsY);
      dim3 grid((N + gemm_int8::kBlockN - 1) / gemm_int8::kBlockN,
                (M + gemm_int8::kBlockM - 1) / gemm_int8::kBlockM);

      gemm_int8::GemmINT8Kernel<<<grid, block, gemm_int8::kSmemSize>>>(dA, dB, dC, M, N, K);
      CHECK_CUDA(cudaDeviceSynchronize());

      std::vector<int32_t> C_int32(C_gpu_fp32.size());
      CHECK_CUDA(cudaMemcpy(C_int32.data(), dC, C_int32.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
      for (size_t j = 0; j < C_int32.size(); ++j) {
        C_gpu_fp32[j] = static_cast<float>(C_int32[j]) * scale_c;
      }

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

      CHECK_CUDA(cudaEventDestroy(start));
      CHECK_CUDA(cudaEventDestroy(stop));
      CHECK_CUDA(cudaFree(dA));
      CHECK_CUDA(cudaFree(dB));
      CHECK_CUDA(cudaFree(dC));
    }

    bool ok = true;
    double max_abs_diff = 0.0;
    const char* check = "SKIP_UNALIGNED";
    if (aligned && M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim) {
      ok = common::CheckEqual(C_cpu, C_gpu_fp32, 1.0f);
      max_abs_diff = common::MaxAbsDiff(C_cpu, C_gpu_fp32);
      check = ok ? "PASS" : "FAIL";
    } else if (aligned) {
      check = "SKIP";
    }

    const double gflops = (gpu_ms > 0.0f) ? (2.0 * M * N * K / (gpu_ms * 1e6)) : 0.0;
    std::cout << M << "x" << N << "x" << K
              << " | " << std::fixed << std::setprecision(4) << gpu_ms << " ms"
              << " | " << std::setprecision(1) << gflops << " GFLOP/s"
              << " | " << check
              << " | qerr=" << std::setprecision(6) << quant_err << "\n";

    ofs << i << ",gemm_int8," << M << "," << N << "," << K << ","
        << gpu_ms << "," << gflops << "," << max_abs_diff << "," << check << ","
        << quant_err << "\n";
  }
  return 0;
}
