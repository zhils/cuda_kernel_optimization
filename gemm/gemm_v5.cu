// GEMM V5: Tensor Core TF32 WMMA implementation.
//
// Built on V4 direction:
// - Keep large CTA tile idea (128x128)
// - Switch compute core to Tensor Core via WMMA (TF32)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

using namespace nvcuda;

namespace {
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;  // FP16 WMMA uses m16n16k16

constexpr int kCtaM = 128;
constexpr int kCtaN = 128;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;  // 256
}  // namespace

__global__ void GemmV5WmmaFp16Kernel(const half* __restrict__ A,
                                     const half* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K) {
  const int warp_id = threadIdx.x / 32;   // [0,7]
  const int lane_id = threadIdx.x % 32;
  (void)lane_id;

  const int block_row = blockIdx.y * kCtaM;
  const int block_col = blockIdx.x * kCtaN;

  constexpr int kTileMCount = kCtaM / kWmmaM;  // 8
  constexpr int kTileNCount = kCtaN / kWmmaN;  // 8
  constexpr int kTilesPerBlock = kTileMCount * kTileNCount;  // 64

  for (int tile = warp_id; tile < kTilesPerBlock; tile += kWarpsPerBlock) {
    const int tile_m = tile / kTileNCount;
    const int tile_n = tile % kTileNCount;
    const int row = block_row + tile_m * kWmmaM;
    const int col = block_col + tile_n * kWmmaN;

    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int kk = 0; kk < K; kk += kWmmaK) {
      const half* a_ptr = A + row * K + kk;
      const half* b_ptr = B + kk * N + col;

      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                     half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                     half, wmma::row_major> b_frag;

      wmma::load_matrix_sync(a_frag, a_ptr, K);
      wmma::load_matrix_sync(b_frag, b_ptr, N);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
  }
}

static void GemmCPU(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < N; ++c) {
      float s = 0;
      for (int k = 0; k < K; ++k) s += A[r * K + k] * B[k * N + c];
      C[r * N + c] = s;
    }
}

int main() {
  constexpr int kRepeat = 10;
  constexpr int kMaxCpuVerifyDim = 1024;
  constexpr int kMaxGpuRunDim = 4096;

  auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/gemm_v5_results.csv");
  ofs << "id,group,M,N,K,gpu_ms,gflops,max_abs_diff,check\n";

  for (size_t i = 0; i < cases.size(); ++i) {
    int M = cases[i].rows, N = cases[i].cols, K = M;
    const bool do_gpu_run = (M <= kMaxGpuRunDim && N <= kMaxGpuRunDim && K <= kMaxGpuRunDim);
    const bool aligned = (M % kCtaM == 0) && (N % kCtaN == 0) && (K % kWmmaK == 0);
    std::vector<float> A(static_cast<size_t>(M) * K),
                       B(static_cast<size_t>(K) * N),
                       cpu(static_cast<size_t>(M) * N),
                       gpu(static_cast<size_t>(M) * N);
    std::vector<half> A_half(static_cast<size_t>(M) * K),
                      B_half(static_cast<size_t>(K) * N);
    common::InitMatrix(A, M, K);
    common::InitMatrix(B, K, N);
    for (size_t idx = 0; idx < A.size(); ++idx) A_half[idx] = __float2half(A[idx]);
    for (size_t idx = 0; idx < B.size(); ++idx) B_half[idx] = __float2half(B[idx]);

    const bool do_cpu_verify =
        (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim && K <= kMaxCpuVerifyDim);
    if (do_cpu_verify) {
      GemmCPU(A.data(), B.data(), cpu.data(), M, N, K);
    }

    float gpu_ms = 0.0f;
    if (do_gpu_run && aligned) {
      half *dA, *dB;
      float* dC;
      CHECK_CUDA(cudaMalloc(&dA, A_half.size() * sizeof(half)));
      CHECK_CUDA(cudaMalloc(&dB, B_half.size() * sizeof(half)));
      CHECK_CUDA(cudaMalloc(&dC, gpu.size() * sizeof(float)));
      CHECK_CUDA(cudaMemcpy(dA, A_half.data(), A_half.size() * sizeof(half), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(dB, B_half.data(), B_half.size() * sizeof(half), cudaMemcpyHostToDevice));

      dim3 block(kThreadsPerBlock);
      dim3 grid((N + kCtaN - 1) / kCtaN, (M + kCtaM - 1) / kCtaM);

      GemmV5WmmaFp16Kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
      CHECK_CUDA(cudaDeviceSynchronize());

      cudaEvent_t s, e;
      CHECK_CUDA(cudaEventCreate(&s));
      CHECK_CUDA(cudaEventCreate(&e));
      std::vector<float> gpu_times;
      gpu_times.reserve(kRepeat);
      for (int rep = 0; rep < kRepeat; ++rep) {
        CHECK_CUDA(cudaEventRecord(s));
        GemmV5WmmaFp16Kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));
        CHECK_CUDA(cudaGetLastError());
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
        gpu_times.push_back(ms);
      }
      std::sort(gpu_times.begin(), gpu_times.end());
      if (gpu_times.size() > 2) {
        for (size_t t = 1; t + 1 < gpu_times.size(); ++t) gpu_ms += gpu_times[t];
        gpu_ms /= static_cast<float>(gpu_times.size() - 2);
      } else if (!gpu_times.empty()) {
        for (float t : gpu_times) gpu_ms += t;
        gpu_ms /= static_cast<float>(gpu_times.size());
      }

      CHECK_CUDA(cudaMemcpy(gpu.data(), dC, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaEventDestroy(s));
      CHECK_CUDA(cudaEventDestroy(e));
      CHECK_CUDA(cudaFree(dA));
      CHECK_CUDA(cudaFree(dB));
      CHECK_CUDA(cudaFree(dC));
    }

    bool ok = true;
    double max_abs_diff = 0.0;
    const char* check = "SKIP";
    if (!do_gpu_run) {
      check = "SKIP_GPU_LARGE";
    } else if (!aligned) {
      check = "SKIP_UNALIGNED";
    } else if (do_cpu_verify) {
      // FP16 input with FP32 accumulate: keep a relaxed threshold for fair check.
      ok = common::CheckEqual(cpu, gpu, 5e-2f);
      max_abs_diff = common::MaxAbsDiff(cpu, gpu);
      check = ok ? "PASS" : "FAIL";
    }
    double gflops = (gpu_ms > 0.0f) ? (2.0 * M * N * K / (gpu_ms * 1e6)) : 0.0;

    std::cout << "M=" << M << " N=" << N << " K=" << K
              << " | " << std::fixed << std::setprecision(3) << gpu_ms << " ms"
              << " | " << std::setprecision(1) << gflops << " GFLOPS"
              << " | " << check << "\n";

    ofs << cases[i].id << "," << cases[i].group << "," << M << "," << N << "," << K << ","
        << gpu_ms << "," << gflops << "," << max_abs_diff << "," << check << "\n";
  }
  return 0;
}
