#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"
// inline CpuGemm defined below

static void GemmCPU(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double sum = 0;
      for (int k = 0; k < K; ++k) sum += static_cast<double>(A[i * K + k]) * B[k * N + j];
      C[i * N + j] = static_cast<float>(sum);
    }
}

namespace gemm_v4 {
namespace wmma = nvcuda::wmma;

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kTileK = 16;

constexpr int kNumWarpsM = 2;
constexpr int kNumWarpsN = 4;
constexpr int kWarpSize = 32;
constexpr int kThreads = kNumWarpsM * kNumWarpsN * kWarpSize;  // 256
static_assert(kThreads == 256, "thread count");

constexpr int kWarpTilesM = 4;
constexpr int kWarpTilesN = 2;
constexpr int kWarpM = kWarpTilesM * 16;  // 64
constexpr int kWarpN = kWarpTilesN * 16;  // 32
static_assert(kNumWarpsM * kWarpM == kBlockM, "block M coverage");
static_assert(kNumWarpsN * kWarpN == kBlockN, "block N coverage");

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 8;
static_assert(kTileK % kWmmaK == 0, "kTileK must be multiple of kWmmaK");

constexpr int kBlockThreadsX = 16;
constexpr int kBlockThreadsY = 16;

constexpr int kSmemABuf = kBlockM * kTileK;
constexpr int kSmemBBuf = kTileK * kBlockN;
constexpr size_t kSmemSize = sizeof(float) * 2 * (kSmemABuf + kSmemBBuf);

__global__ __launch_bounds__(kThreads, 2) void GemmV4Kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
  // 共享内存双缓冲
  extern __shared__ float shared_mem[];
  float* As_buf0 = shared_mem;
  float* As_buf1 = shared_mem + kSmemABuf;
  float* Bs_buf0 = shared_mem + 2 * kSmemABuf;
  float* Bs_buf1 = shared_mem + 2 * kSmemABuf + kSmemBBuf;

  // 线程与warp索引
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * kBlockThreadsX + tx;
  const int warp_id = tid / kWarpSize;
  const int warp_m = warp_id / kNumWarpsN;
  const int warp_n = warp_id % kNumWarpsN;

  // 初始化WMMA累加器
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

  // 异步加载函数：cp.async 加载 A/B 块
  auto load_tile_async = [&](int tile_k_start, float* As_buf, float* Bs_buf) {
    const int total_a_float4 = (kBlockM * kTileK) / 4;
    const int a_float4_per_thread = (total_a_float4 + kThreads - 1) / kThreads;
    for (int l = 0; l < a_float4_per_thread; ++l) {
      int idx = tid * a_float4_per_thread + l;
      if (idx >= total_a_float4) continue;
      int r = idx / (kTileK / 4);
      int k_offset = (idx % (kTileK / 4)) * 4;
      int g_r = blockIdx.y * kBlockM + r;
      int g_k = tile_k_start + k_offset;
      float* dst = As_buf + r * kTileK + k_offset;
      if (g_r < M && g_k + 3 < K) {
        __pipeline_memcpy_async(
            dst,
            &A[static_cast<size_t>(g_r) * K + g_k], 16);
      } else {
        dst[0] = (g_r < M && g_k + 0 < K) ? A[static_cast<size_t>(g_r) * K + g_k + 0] : 0.0f;
        dst[1] = (g_r < M && g_k + 1 < K) ? A[static_cast<size_t>(g_r) * K + g_k + 1] : 0.0f;
        dst[2] = (g_r < M && g_k + 2 < K) ? A[static_cast<size_t>(g_r) * K + g_k + 2] : 0.0f;
        dst[3] = (g_r < M && g_k + 3 < K) ? A[static_cast<size_t>(g_r) * K + g_k + 3] : 0.0f;
      }
    }

    const int total_b_float4 = (kTileK * kBlockN) / 4;
    const int b_float4_per_thread = (total_b_float4 + kThreads - 1) / kThreads;
    for (int l = 0; l < b_float4_per_thread; ++l) {
      int idx = tid * b_float4_per_thread + l;
      if (idx >= total_b_float4) continue;
      int k_idx = idx / (kBlockN / 4);
      int c_offset = (idx % (kBlockN / 4)) * 4;
      int g_k = tile_k_start + k_idx;
      int g_c = blockIdx.x * kBlockN + c_offset;
      float* dst = Bs_buf + k_idx * kBlockN + c_offset;
      if (g_k < K && g_c + 3 < N) {
        __pipeline_memcpy_async(
            dst,
            &B[static_cast<size_t>(g_k) * N + g_c], 16);
      } else {
        dst[0] = (g_k < K && g_c + 0 < N) ? B[static_cast<size_t>(g_k) * N + g_c + 0] : 0.0f;
        dst[1] = (g_k < K && g_c + 1 < N) ? B[static_cast<size_t>(g_k) * N + g_c + 1] : 0.0f;
        dst[2] = (g_k < K && g_c + 2 < N) ? B[static_cast<size_t>(g_k) * N + g_c + 2] : 0.0f;
        dst[3] = (g_k < K && g_c + 3 < N) ? B[static_cast<size_t>(g_k) * N + g_c + 3] : 0.0f;
      }
    }
  };

  // 加载首块并同步
  load_tile_async(0, As_buf0, Bs_buf0);
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();

  // 主循环：异步加载下一块 + WMMA计算当前块
  #pragma unroll
  for (int t = 0; t < num_k_tiles; ++t) {
    float* As_read = (t & 1) ? As_buf1 : As_buf0;
    float* Bs_read = (t & 1) ? Bs_buf1 : Bs_buf0;

    if (t + 1 < num_k_tiles) {
      float* As_write = (t & 1) ? As_buf0 : As_buf1;
      float* Bs_write = (t & 1) ? Bs_buf0 : Bs_buf1;
      load_tile_async((t + 1) * kTileK, As_write, Bs_write);
      __pipeline_commit();
    }

    // WMMA计算：加载矩阵片段 + 执行乘加
    #pragma unroll
    for (int kk = 0; kk < kTileK; kk += kWmmaK) {
      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                     wmma::precision::tf32, wmma::row_major>
          a_frag[kWarpTilesM];
      #pragma unroll
      for (int i = 0; i < kWarpTilesM; ++i) {
        wmma::load_matrix_sync(a_frag[i],
            As_read + (a_offset_base + i * kWmmaM) * a_ld + kk, a_ld);
      }

      #pragma unroll
      for (int j = 0; j < kWarpTilesN; ++j) {
        wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                       wmma::precision::tf32, wmma::row_major>
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

  // 写回结果
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

}  // namespace gemm_v4

#ifndef ALL_COMPARE_LIB
int main() {
  // 参数与输出文件准备
  constexpr int kRepeat = 10;
  constexpr int kMaxCpuVerifyDim = 1024;
  auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/gemm_v4_results.csv");
  ofs << "id,group,M,N,K,gpu_ms,gflops,max_abs_diff,check\n";

  CHECK_CUDA(cudaFuncSetAttribute(gemm_v4::GemmV4Kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      gemm_v4::kSmemSize));

  for (size_t i = 0; i < cases.size(); ++i) {
    const int M = cases[i].rows;
    const int N = cases[i].cols;
    const int K = M;
    const bool aligned = (M % gemm_v4::kBlockM == 0) &&
                         (N % gemm_v4::kBlockN == 0) &&
                         (K % gemm_v4::kTileK == 0);

    // 生成测试数据
    std::vector<float> A(static_cast<size_t>(M) * K),
                       B(static_cast<size_t>(K) * N),
                       C_cpu(static_cast<size_t>(M) * N),
                       C_gpu(static_cast<size_t>(M) * N);
    common::InitMatrix(A, M, K);
    common::InitMatrix(B, K, N);

    // CPU参考计算
    if (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim) {
      GemmCPU(A.data(), B.data(), C_cpu.data(), M, N, K);
    }

    float gpu_ms = 0.0f;
    if (aligned) {
      // 分配GPU内存
      float *dA, *dB, *dC;
      CHECK_CUDA(cudaMalloc(&dA, A.size() * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&dB, B.size() * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&dC, C_gpu.size() * sizeof(float)));

      // 拷贝数据到设备
      CHECK_CUDA(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

      // 启动配置
      dim3 block(gemm_v4::kBlockThreadsX, gemm_v4::kBlockThreadsY);
      dim3 grid((N + gemm_v4::kBlockN - 1) / gemm_v4::kBlockN,
                (M + gemm_v4::kBlockM - 1) / gemm_v4::kBlockM);

      // 预热
      gemm_v4::GemmV4Kernel<<<grid, block, gemm_v4::kSmemSize>>>(dA, dB, dC, M, N, K);
      CHECK_CUDA(cudaDeviceSynchronize());

      // 计时循环
      cudaEvent_t start, stop;
      CHECK_CUDA(cudaEventCreate(&start));
      CHECK_CUDA(cudaEventCreate(&stop));
      CHECK_CUDA(cudaEventRecord(start));
      for (int rep = 0; rep < kRepeat; ++rep) {
        gemm_v4::GemmV4Kernel<<<grid, block, gemm_v4::kSmemSize>>>(dA, dB, dC, M, N, K);
      }
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));
      CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));
      gpu_ms /= static_cast<float>(kRepeat);

      // 拷贝结果回主机
      CHECK_CUDA(cudaMemcpy(C_gpu.data(), dC, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

      // 释放GPU资源
      CHECK_CUDA(cudaEventDestroy(start));
      CHECK_CUDA(cudaEventDestroy(stop));
      CHECK_CUDA(cudaFree(dA));
      CHECK_CUDA(cudaFree(dB));
      CHECK_CUDA(cudaFree(dC));
    }

    // 校验
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

    // 计算并输出结果
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

#endif /* ALL_COMPARE_LIB */
