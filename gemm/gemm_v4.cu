#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

namespace gemm_v4 {
namespace wmma = nvcuda::wmma;

constexpr int kThreads = 128;
constexpr int kBlockThreadsX = 16;
constexpr int kBlockThreadsY = 8;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 8;
constexpr int kWarpTilesM = 2;
constexpr int kWarpTilesN = 2;
constexpr int kBlockM = kWarpTilesM * kWmmaM;  // 32
constexpr int kBlockN = kWarpTilesN * kWmmaN;  // 32
constexpr int kTileK = 16;

static_assert(kWarpsPerBlock == kWarpTilesM * kWarpTilesN, "warp mapping mismatch");

__shared__ float As[2][kBlockM][kTileK];
__shared__ float Bs[2][kTileK][kBlockN];

__global__ __launch_bounds__(kThreads, 2) void GemmV4Kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * kBlockThreadsX + tx;
  const int warp_id = tid / kWarpSize;
  const int warp_m = warp_id / kWarpTilesN;
  const int warp_n = warp_id % kWarpTilesN;

  const int num_k_tiles = (K + kTileK - 1) / kTileK;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  auto load_tile = [&](int tile_idx, int buf) {
    const int k0 = tile_idx * kTileK;
    const int total_a_float4 = (kBlockM * kTileK) / 4;
    const int a_float4_per_thread = (total_a_float4 + kThreads - 1) / kThreads;
    for (int l = 0; l < a_float4_per_thread; ++l) {
      const int idx = tid * a_float4_per_thread + l;
      if (idx >= total_a_float4) continue;
      const int r = idx / (kTileK / 4);
      const int k_offset = (idx % (kTileK / 4)) * 4;
      const int g_r = blockIdx.y * kBlockM + r;
      const int g_k = k0 + k_offset;
      if (g_r < M && g_k + 3 < K) {
        const float4 v = __ldg(reinterpret_cast<const float4*>(A + static_cast<size_t>(g_r) * K + g_k));
        As[buf][r][k_offset + 0] = v.x;
        As[buf][r][k_offset + 1] = v.y;
        As[buf][r][k_offset + 2] = v.z;
        As[buf][r][k_offset + 3] = v.w;
      } else {
        As[buf][r][k_offset + 0] = (g_r < M && g_k + 0 < K) ? A[static_cast<size_t>(g_r) * K + g_k + 0] : 0.0f;
        As[buf][r][k_offset + 1] = (g_r < M && g_k + 1 < K) ? A[static_cast<size_t>(g_r) * K + g_k + 1] : 0.0f;
        As[buf][r][k_offset + 2] = (g_r < M && g_k + 2 < K) ? A[static_cast<size_t>(g_r) * K + g_k + 2] : 0.0f;
        As[buf][r][k_offset + 3] = (g_r < M && g_k + 3 < K) ? A[static_cast<size_t>(g_r) * K + g_k + 3] : 0.0f;
      }
    }

    const int total_b_float4 = (kTileK * kBlockN) / 4;
    const int b_float4_per_thread = (total_b_float4 + kThreads - 1) / kThreads;
    for (int l = 0; l < b_float4_per_thread; ++l) {
      const int idx = tid * b_float4_per_thread + l;
      if (idx >= total_b_float4) continue;
      const int k_idx = idx / (kBlockN / 4);
      const int c_offset = (idx % (kBlockN / 4)) * 4;
      const int g_k = k0 + k_idx;
      const int g_c = blockIdx.x * kBlockN + c_offset;
      if (g_k < K && g_c + 3 < N) {
        const float4 v = __ldg(reinterpret_cast<const float4*>(B + static_cast<size_t>(g_k) * N + g_c));
        Bs[buf][k_idx][c_offset + 0] = v.x;
        Bs[buf][k_idx][c_offset + 1] = v.y;
        Bs[buf][k_idx][c_offset + 2] = v.z;
        Bs[buf][k_idx][c_offset + 3] = v.w;
      } else {
        Bs[buf][k_idx][c_offset + 0] = (g_k < K && g_c + 0 < N) ? B[static_cast<size_t>(g_k) * N + g_c + 0] : 0.0f;
        Bs[buf][k_idx][c_offset + 1] = (g_k < K && g_c + 1 < N) ? B[static_cast<size_t>(g_k) * N + g_c + 1] : 0.0f;
        Bs[buf][k_idx][c_offset + 2] = (g_k < K && g_c + 2 < N) ? B[static_cast<size_t>(g_k) * N + g_c + 2] : 0.0f;
        Bs[buf][k_idx][c_offset + 3] = (g_k < K && g_c + 3 < N) ? B[static_cast<size_t>(g_k) * N + g_c + 3] : 0.0f;
      }
    }
  };

  load_tile(0, 0);
  __syncthreads();

  int read_buf = 0;
  int write_buf = 1;
  for (int t = 0; t < num_k_tiles; ++t) {
    if (t + 1 < num_k_tiles) {
      load_tile(t + 1, write_buf);
    }

    #pragma unroll
    for (int kk = 0; kk < kTileK; kk += kWmmaK) {
      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, wmma::precision::tf32, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, wmma::precision::tf32, wmma::row_major> b_frag;
      const float* tile_a = &As[read_buf][warp_m * kWmmaM][kk];
      const float* tile_b = &Bs[read_buf][kk][warp_n * kWmmaN];
      wmma::load_matrix_sync(a_frag, tile_a, kTileK);
      wmma::load_matrix_sync(b_frag, tile_b, kBlockN);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __syncthreads();
    int tmp = read_buf;
    read_buf = write_buf;
    write_buf = tmp;
  }

  const int out_r = blockIdx.y * kBlockM + warp_m * kWmmaM;
  const int out_c = blockIdx.x * kBlockN + warp_n * kWmmaN;
  if (out_r + kWmmaM <= M && out_c + kWmmaN <= N) {
    wmma::store_matrix_sync(C + static_cast<size_t>(out_r) * N + out_c, c_frag, N, wmma::mem_row_major);
  }
}

}  // namespace gemm_v4

static void GemmCPU(const float* A, const float* B, float* C, int M, int N, int K) {
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
  std::ofstream ofs("data/results/gemm_v4_results.csv");
  ofs << "id,group,M,N,K,gpu_ms,gflops,max_abs_diff,check\n";

  for (size_t i = 0; i < cases.size(); ++i) {
    const int M = cases[i].rows;
    const int N = cases[i].cols;
    const int K = M;
    const bool aligned = (M % gemm_v4::kBlockM == 0) && (N % gemm_v4::kBlockN == 0) &&
                         (K % gemm_v4::kTileK == 0);
    std::vector<float> A(static_cast<size_t>(M) * K),
                       B(static_cast<size_t>(K) * N),
                       C_cpu(static_cast<size_t>(M) * N),
                       C_gpu(static_cast<size_t>(M) * N);
    common::InitMatrix(A, M, K);
    common::InitMatrix(B, K, N);

    if (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim) {
      GemmCPU(A.data(), B.data(), C_cpu.data(), M, N, K);
    }

    float gpu_ms = 0.0f;
    if (aligned) {
      float *dA, *dB, *dC;
      CHECK_CUDA(cudaMalloc(&dA, A.size() * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&dB, B.size() * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&dC, C_gpu.size() * sizeof(float)));
      CHECK_CUDA(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

      dim3 block(gemm_v4::kBlockThreadsX, gemm_v4::kBlockThreadsY);
      dim3 grid((N + gemm_v4::kBlockN - 1) / gemm_v4::kBlockN,
                (M + gemm_v4::kBlockM - 1) / gemm_v4::kBlockM);
      gemm_v4::GemmV4Kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
      CHECK_CUDA(cudaDeviceSynchronize());

      cudaEvent_t start, stop;
      CHECK_CUDA(cudaEventCreate(&start));
      CHECK_CUDA(cudaEventCreate(&stop));
      CHECK_CUDA(cudaEventRecord(start));
      for (int rep = 0; rep < kRepeat; ++rep) {
        gemm_v4::GemmV4Kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
      }
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));
      gpu_ms /= static_cast<float>(kRepeat);

      CHECK_CUDA(cudaMemcpy(C_gpu.data(), dC, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
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
      ok = common::CheckEqual(C_cpu, C_gpu, 2e-1f);
      max_abs_diff = common::MaxAbsDiff(C_cpu, C_gpu);
      check = ok ? "PASS" : "FAIL";
    } else if (aligned) {
      check = "SKIP";
    }

    const double gflops = (gpu_ms > 0.0f) ? (2.0 * M * N * K / (gpu_ms * 1e6)) : 0.0;
    std::cout << M << "x" << N << "x" << K
              << " | " << std::fixed << std::setprecision(4) << gpu_ms << " ms"
              << " | " << std::setprecision(1) << gflops << " GFLOP/s"
              << " | " << check << "\n";

    ofs << i << ",gemm_v4," << M << "," << N << "," << K << ","
        << gpu_ms << "," << gflops << "," << max_abs_diff << "," << check << "\n";
  }
  return 0;
}
