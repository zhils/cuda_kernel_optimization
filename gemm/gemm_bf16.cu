// GEMM BF16: BF16 WMMA Tensor Core (k=16), FP32 output
//
// Blackwell sm_120 FP16/BF16 TC: 256 TC lanes x 2 FMA/s cycle x 2550 MHz
//   per-SM peak: ~1.3 TFLOPS, 36 SM => ~47 TFLOPS sustained estimate
//
// Structure mirrors gemm_fp16.cu:
//   - cp.async pipeline (16-byte aligned)
//   - nvcuda::wmma::mma_sync (m16n16k16)
//   - FP32 accumulation for precision

#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cuda_bf16.h>
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

namespace gemm_bf16 {
namespace wmma = nvcuda::wmma;

constexpr int kBlockM = 128, kBlockN = 128, kTileK = 32;
constexpr int kNumWarpsM = 2, kNumWarpsN = 4, kWarpSize = 32;
constexpr int kThreads = kNumWarpsM * kNumWarpsN * kWarpSize;  // 256
constexpr int kWarpTilesM = 4, kWarpTilesN = 2;
constexpr int kWarpM = kWarpTilesM * 16, kWarpN = kWarpTilesN * 16;
constexpr int kWmmaM = 16, kWmmaN = 16, kWmmaK = 16;
constexpr int kBlockThreadsX = 16, kBlockThreadsY = 16;
constexpr int kSmemABuf = kBlockM * kTileK;
constexpr int kSmemBBuf = kTileK * kBlockN;
constexpr size_t kSmemSize = sizeof(__nv_bfloat16) * 2 * (kSmemABuf + kSmemBBuf);  // 32 KB

__global__ __launch_bounds__(kThreads, 2) void GemmBF16Kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {
  extern __shared__ __nv_bfloat16 shared_mem[];
  __nv_bfloat16* As_buf0 = shared_mem;
  __nv_bfloat16* As_buf1 = shared_mem + kSmemABuf;
  __nv_bfloat16* Bs_buf0 = shared_mem + 2 * kSmemABuf;
  __nv_bfloat16* Bs_buf1 = shared_mem + 2 * kSmemABuf + kSmemBBuf;
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int tid = ty * kBlockThreadsX + tx;
  const int warp_id = tid / kWarpSize;
  const int warp_m = warp_id / kNumWarpsN;
  const int warp_n = warp_id % kNumWarpsN;

  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>
      c_frag[kWarpTilesM][kWarpTilesN];
  #pragma unroll
  for (int i = 0; i < kWarpTilesM; ++i)
    for (int j = 0; j < kWarpTilesN; ++j)
      wmma::fill_fragment(c_frag[i][j], 0.0f);

  const int num_k_tiles = (K + kTileK - 1) / kTileK;
  const int a_off = warp_m * kWarpM;
  const int b_off = warp_n * kWarpN;

  auto load_tile = [&](int k_start, __nv_bfloat16* As, __nv_bfloat16* Bs) {
    const int total_a8 = (kBlockM * kTileK) / 8;
    const int a_per = (total_a8 + kThreads - 1) / kThreads;
    for (int l = 0; l < a_per; ++l) {
      int idx = tid * a_per + l;
      if (idx >= total_a8) continue;
      int r = idx / (kTileK / 8), ko = (idx % (kTileK / 8)) * 8;
      int gr = blockIdx.y * kBlockM + r, gk = k_start + ko;
      __nv_bfloat16* dst = As + r * kTileK + ko;
      if (gr < M && gk + 7 < K)
        __pipeline_memcpy_async(dst, &A[size_t(gr) * K + gk], 16);
      else
        for (int i = 0; i < 8; ++i)
          dst[i] = (gr < M && gk + i < K) ? A[size_t(gr) * K + gk + i] : __nv_bfloat16(0);
    }
    const int total_b8 = (kTileK * kBlockN) / 8;
    const int b_per = (total_b8 + kThreads - 1) / kThreads;
    for (int l = 0; l < b_per; ++l) {
      int idx = tid * b_per + l;
      if (idx >= total_b8) continue;
      int ki = idx / (kBlockN / 8), co = (idx % (kBlockN / 8)) * 8;
      int gk = k_start + ki, gc = blockIdx.x * kBlockN + co;
      __nv_bfloat16* dst = Bs + ki * kBlockN + co;
      if (gk < K && gc + 7 < N)
        __pipeline_memcpy_async(dst, &B[size_t(gk) * N + gc], 16);
      else
        for (int i = 0; i < 8; ++i)
          dst[i] = (gk < K && gc + i < N) ? B[size_t(gk) * N + gc + i] : __nv_bfloat16(0);
    }
  };

  load_tile(0, As_buf0, Bs_buf0);
  __pipeline_commit(); __pipeline_wait_prior(0); __syncthreads();

  for (int t = 0; t < num_k_tiles; ++t) {
    __nv_bfloat16* As_r = (t & 1) ? As_buf1 : As_buf0;
    __nv_bfloat16* Bs_r = (t & 1) ? Bs_buf1 : Bs_buf0;
    if (t + 1 < num_k_tiles) {
      load_tile((t + 1) * kTileK, (t & 1) ? As_buf0 : As_buf1, (t & 1) ? Bs_buf0 : Bs_buf1);
      __pipeline_commit();
    }
    for (int kk = 0; kk < kTileK; kk += kWmmaK) {
      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::row_major>
          a_frag[kWarpTilesM];
      for (int i = 0; i < kWarpTilesM; ++i)
        wmma::load_matrix_sync(a_frag[i], As_r + (a_off + i * kWmmaM) * kTileK + kk, kTileK);
      for (int j = 0; j < kWarpTilesN; ++j) {
        wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::row_major> b_frag;
        wmma::load_matrix_sync(b_frag, Bs_r + kk * kBlockN + b_off + j * kWmmaN, kBlockN);
        for (int i = 0; i < kWarpTilesM; ++i)
          wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag, c_frag[i][j]);
      }
    }
    if (t + 1 < num_k_tiles) __pipeline_wait_prior(0);
    __syncthreads();
  }

  const int out_r = blockIdx.y * kBlockM + a_off;
  const int out_c = blockIdx.x * kBlockN + b_off;
  for (int i = 0; i < kWarpTilesM; ++i)
    for (int j = 0; j < kWarpTilesN; ++j) {
      int gr = out_r + i * kWmmaM, gc = out_c + j * kWmmaN;
      if (gr + kWmmaM <= M && gc + kWmmaN <= N)
        wmma::store_matrix_sync(C + size_t(gr) * N + gc, c_frag[i][j], N, wmma::mem_row_major);
    }
}
}  // namespace gemm_bf16

// --- Precision metrics (shared with gemm_fp16) ---
struct PrecisionMetrics {
  double cos_sim, snr_db, max_rel_err, mean_abs_err, p99_abs_err, max_abs_err;
};

static PrecisionMetrics ComputeMetrics(const float* ref, const float* test, size_t n) {
  PrecisionMetrics m = {};
  double dot = 0, nr = 0, nt = 0, npow = 0, spow = 0, max_rel = 0, sum_abs = 0, max_abs = 0;
  std::vector<double> abs_errs(n);
  double ref_max = 0;
  for (size_t i = 0; i < n; ++i) ref_max = std::max(ref_max, double(std::fabs(ref[i])));
  double rt = std::max(ref_max * 1e-6, 1e-6);
  for (size_t i = 0; i < n; ++i) {
    double r = ref[i], t = test[i];
    dot += r * t; nr += r * r; nt += t * t;
    double d = r - t; npow += d * d; spow += r * r;
    double ab = std::fabs(d); abs_errs[i] = ab; sum_abs += ab;
    max_abs = std::max(max_abs, ab);
    if (std::fabs(r) >= rt) max_rel = std::max(max_rel, ab / std::fabs(r));
  }
  m.cos_sim = dot / (std::sqrt(nr) * std::sqrt(nt) + 1e-10);
  m.snr_db = 10.0 * std::log10(spow / (npow + 1e-10));
  m.max_rel_err = max_rel; m.mean_abs_err = sum_abs / n; m.max_abs_err = max_abs;
  if (n > 0) {
    size_t p = std::min(size_t(n * 0.99), n - 1);
    std::nth_element(abs_errs.begin(), abs_errs.begin() + p, abs_errs.end());
    m.p99_abs_err = abs_errs[p];
  }
  return m;
}

static void GemmCPU_FP32(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < N; ++c) {
      float s = 0;
      for (int k = 0; k < K; ++k) s += A[size_t(r) * K + k] * B[size_t(k) * N + c];
      C[size_t(r) * N + c] = s;
    }
}

#ifndef ALL_COMPARE_LIB
int main() {
  constexpr int kRepeat = 10, kMaxCpu = 1024;
  auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/gemm_bf16_results.csv");
  ofs << "id,group,M,N,K,gpu_ms,gflops,cos_sim,snr_db,max_rel_err,mean_abs_err,p99_abs_err,max_abs_err\n";
  cudaFuncSetAttribute(gemm_bf16::GemmBF16Kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, gemm_bf16::kSmemSize);

  for (size_t i = 0; i < cases.size(); ++i) {
    int M = cases[i].rows, N = cases[i].cols, K = M;
    bool aligned = (M % gemm_bf16::kBlockM == 0) && (N % gemm_bf16::kBlockN == 0) && (K % gemm_bf16::kTileK == 0);
    size_t Csz = size_t(M) * N;
    std::vector<float> A_f32(size_t(M)*K), B_f32(size_t(K)*N), C_cpu(Csz), C_gpu(Csz);
    common::InitMatrix(A_f32, M, K); common::InitMatrix(B_f32, K, N);
    if (M <= kMaxCpu && N <= kMaxCpu) GemmCPU_FP32(A_f32.data(), B_f32.data(), C_cpu.data(), M, N, K);
    float ms = 0;
    if (aligned) {
      std::vector<__nv_bfloat16> A_bf(A_f32.size()), B_bf(B_f32.size());
      for (size_t j = 0; j < A_f32.size(); ++j) A_bf[j] = __float2bfloat16(A_f32[j]);
      for (size_t j = 0; j < B_f32.size(); ++j) B_bf[j] = __float2bfloat16(B_f32[j]);
      __nv_bfloat16 *dA, *dB; float *dC;
      cudaMalloc(&dA, A_bf.size() * sizeof(__nv_bfloat16));
      cudaMalloc(&dB, B_bf.size() * sizeof(__nv_bfloat16));
      cudaMalloc(&dC, Csz * sizeof(float));
      cudaMemcpy(dA, A_bf.data(), A_bf.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
      cudaMemcpy(dB, B_bf.data(), B_bf.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
      dim3 block(gemm_bf16::kBlockThreadsX, gemm_bf16::kBlockThreadsY);
      dim3 grid((N + gemm_bf16::kBlockN - 1) / gemm_bf16::kBlockN,
                (M + gemm_bf16::kBlockM - 1) / gemm_bf16::kBlockM);
      gemm_bf16::GemmBF16Kernel<<<grid,block,gemm_bf16::kSmemSize>>>(dA,dB,dC,M,N,K);
      cudaDeviceSynchronize();
      cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s);
      for (int r = 0; r < kRepeat; ++r)
        gemm_bf16::GemmBF16Kernel<<<grid,block,gemm_bf16::kSmemSize>>>(dA,dB,dC,M,N,K);
      cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); ms /= kRepeat;
      cudaMemcpy(C_gpu.data(), dC, Csz * sizeof(float), cudaMemcpyDeviceToHost);
      cudaEventDestroy(s); cudaEventDestroy(e); cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    PrecisionMetrics pm = {};
    if (aligned && M <= kMaxCpu && N <= kMaxCpu) pm = ComputeMetrics(C_cpu.data(), C_gpu.data(), Csz);
    double gflops = (ms > 0) ? (2.0 * M * N * K / (ms * 1e6)) : 0;
    std::cout << M << "x" << N << "x" << K << " | " << std::fixed << std::setprecision(4) << ms << " ms | "
              << std::setprecision(1) << gflops << " GFLOPS | cos_sim=" << pm.cos_sim << "\n";
    ofs << i << ",gemm_bf16," << M << "," << N << "," << K << ","
        << ms << "," << gflops << "," << pm.cos_sim << "," << pm.snr_db << ","
        << pm.max_rel_err << "," << pm.mean_abs_err << "," << pm.p99_abs_err << "," << pm.max_abs_err << "\n";
  }
  return 0;
}

#endif /* ALL_COMPARE_LIB */
