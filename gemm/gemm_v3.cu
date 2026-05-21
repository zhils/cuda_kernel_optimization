// GEMM V3：使用 cp.async + 8×4 子块 + TileK=32。kTN=4 产生 8×4 子块，
// 将累加器从 64 个寄存器减少到 32 个以提高占用率。kBlockN=64 将 B 的共享内存
// 减半至 48KB，允许每 SM 运行 2 个块。kTileK=32 将外层循环从 256 次减少到
// 128 次，同时减少 __syncthreads 调用。cp.async DMA 实现异步加载/计算流水线。
// __launch_bounds__(256,2) 注解为每 SM 2 个块预留寄存器。

#include <cuda_runtime.h>
#include <cuda_pipeline.h>

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

constexpr int kTM = 8;
constexpr int kTN = 4;

constexpr int kBlockThreadsX = 16;
constexpr int kBlockThreadsY = 16;
constexpr int kBlockThreads = kBlockThreadsX * kBlockThreadsY;

constexpr int kBlockM = kBlockThreadsY * kTM;
constexpr int kBlockN = kBlockThreadsX * kTN;

constexpr int kTileK = 32;

constexpr int kSmemABuf = kBlockM * kTileK;
constexpr int kSmemBBuf = kTileK * kBlockN;

__global__ __launch_bounds__(kBlockThreads, 2)
void GemmV3Kernel(const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {
    // 共享内存双缓冲
    extern __shared__ float smem[];
    float* As_buf0 = smem;
    float* As_buf1 = smem + kSmemABuf;
    float* Bs_buf0 = smem + 2 * kSmemABuf;
    float* Bs_buf1 = smem + 2 * kSmemABuf + kSmemBBuf;

    // 线程索引
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kBlockThreadsX + tx;

    const int row_start = blockIdx.y * kBlockM + ty * kTM;
    const int col_start = blockIdx.x * kBlockN + tx * kTN;

    // 初始化累加器
    float sum[kTM][kTN] = {};

    const int num_k_tiles = (K + kTileK - 1) / kTileK;

    // 异步加载函数：cp.async 加载 A/B 块
    auto load_tile_async = [&](int tile_k_start, float* As_buf, float* Bs_buf) {
        const int total_a_f4 = (kBlockM * kTileK) / 4;
        const int a_f4_per_th = (total_a_f4 + kBlockThreads - 1) / kBlockThreads;
        for (int l = 0; l < a_f4_per_th; ++l) {
            int idx = tid * a_f4_per_th + l;
            if (idx >= total_a_f4) continue;
            int r      = idx / (kTileK / 4);
            int k_off  = (idx % (kTileK / 4)) * 4;
            int g_r    = blockIdx.y * kBlockM + r;
            int g_k    = tile_k_start + k_off;
            float* dst = As_buf + r * kTileK + k_off;
            if (g_r < M && g_k + 3 < K)
                __pipeline_memcpy_async(dst, &A[static_cast<size_t>(g_r) * K + g_k], 16);
            else {
                dst[0] = (g_r < M && g_k+0 < K) ? A[g_r * K + g_k+0] : 0.0f;
                dst[1] = (g_r < M && g_k+1 < K) ? A[g_r * K + g_k+1] : 0.0f;
                dst[2] = (g_r < M && g_k+2 < K) ? A[g_r * K + g_k+2] : 0.0f;
                dst[3] = (g_r < M && g_k+3 < K) ? A[g_r * K + g_k+3] : 0.0f;
            }
        }
        const int total_b_f4 = (kTileK * kBlockN) / 4;
        const int b_f4_per_th = (total_b_f4 + kBlockThreads - 1) / kBlockThreads;
        for (int l = 0; l < b_f4_per_th; ++l) {
            int idx = tid * b_f4_per_th + l;
            if (idx >= total_b_f4) continue;
            int k_idx   = idx / (kBlockN / 4);
            int c_off   = (idx % (kBlockN / 4)) * 4;
            int g_k     = tile_k_start + k_idx;
            int g_c     = blockIdx.x * kBlockN + c_off;
            float* dst  = Bs_buf + k_idx * kBlockN + c_off;
            if (g_k < K && g_c + 3 < N)
                __pipeline_memcpy_async(dst, &B[static_cast<size_t>(g_k) * N + g_c], 16);
            else {
                dst[0] = (g_k < K && g_c+0 < N) ? B[g_k * N + g_c+0] : 0.0f;
                dst[1] = (g_k < K && g_c+1 < N) ? B[g_k * N + g_c+1] : 0.0f;
                dst[2] = (g_k < K && g_c+2 < N) ? B[g_k * N + g_c+2] : 0.0f;
                dst[3] = (g_k < K && g_c+3 < N) ? B[g_k * N + g_c+3] : 0.0f;
            }
        }
    };

    // 加载首块并同步
    load_tile_async(0, As_buf0, Bs_buf0);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // 主循环：异步加载下一块 + 计算当前块
    for (int t = 0; t < num_k_tiles; ++t) {
        float* As_read = (t & 1) ? As_buf1 : As_buf0;
        float* Bs_read = (t & 1) ? Bs_buf1 : Bs_buf0;

        if (t + 1 < num_k_tiles) {
            float* As_write = (t & 1) ? As_buf0 : As_buf1;
            float* Bs_write = (t & 1) ? Bs_buf0 : Bs_buf1;
            load_tile_async((t + 1) * kTileK, As_write, Bs_write);
            __pipeline_commit();
        }

        // 计算当前块：B值预取 + A遍历
        #pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            float b_vals[kTN];
            #pragma unroll
            for (int j = 0; j < kTN; ++j)
                b_vals[j] = Bs_read[kk * kBlockN + tx * kTN + j];

            #pragma unroll
            for (int i = 0; i < kTM; ++i) {
                float a_val = As_read[(ty * kTM + i) * kTileK + kk];
                #pragma unroll
                for (int j = 0; j < kTN; ++j)
                    sum[i][j] += a_val * b_vals[j];
            }
        }

        if (t + 1 < num_k_tiles) {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
    }

    // 写回结果
    #pragma unroll
    for (int i = 0; i < kTM; ++i) {
        int g_r = row_start + i;
        if (g_r >= M) continue;

        int g_c = col_start;
        if (g_c + 3 < N) {
            float4 v = {sum[i][0], sum[i][1], sum[i][2], sum[i][3]};
            *reinterpret_cast<float4*>(C + g_r * N + g_c) = v;
        } else {
            for (int j = 0; j < kTN && g_c + j < N; ++j)
                C[g_r * N + g_c + j] = sum[i][j];
        }
    }
}

#ifndef ALL_COMPARE_LIB
int main() {
    // 参数与输出文件准备
    constexpr int kRepeat = 10;
    constexpr int kMaxCpuVerifyDim = 1024;
    auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/gemm_v3_results.csv");
    ofs << "id,group,M,N,K,gpu_ms,gflops,max_abs_diff,check\n";

    int smem_bytes = (2 * kSmemABuf + 2 * kSmemBBuf) * sizeof(float);

    for (size_t i = 0; i < cases.size(); ++i) {
        int M = cases[i].rows, N = cases[i].cols, K = M;
        bool aligned = (M % kBlockM == 0) && (N % kBlockN == 0) && (K % kTileK == 0);

        size_t sz_a = static_cast<size_t>(M) * K;
        size_t sz_b = static_cast<size_t>(K) * N;
        size_t sz_c = static_cast<size_t>(M) * N;

        // 生成测试数据
        std::vector<float> A(sz_a), B(sz_b), C_cpu(sz_c), C_gpu(sz_c);
        common::InitMatrix(A, M, K);
        common::InitMatrix(B, K, N);

        // CPU参考计算
        if (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim)
            GemmCPU(A.data(), B.data(), C_cpu.data(), M, N, K);

        float gpu_ms = 0.0f;
        if (aligned) {
            // 分配GPU内存
            float *dA, *dB, *dC;
            CHECK_CUDA(cudaMalloc(&dA, sz_a * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&dB, sz_b * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&dC, sz_c * sizeof(float)));

            // 拷贝数据到设备
            CHECK_CUDA(cudaMemcpy(dA, A.data(), sz_a * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(dB, B.data(), sz_b * sizeof(float), cudaMemcpyHostToDevice));

            // 启动配置
            dim3 block(kBlockThreadsX, kBlockThreadsY);
            dim3 grid((N + kBlockN - 1) / kBlockN, (M + kBlockM - 1) / kBlockM);
            CHECK_CUDA(cudaFuncSetAttribute(GemmV3Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

            // 预热
            GemmV3Kernel<<<grid, block, smem_bytes>>>(dA, dB, dC, M, N, K);
            CHECK_CUDA(cudaDeviceSynchronize());

            // 计时循环
            cudaEvent_t start, stop;
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));
            CHECK_CUDA(cudaEventRecord(start));
            for (int r = 0; r < kRepeat; ++r)
                GemmV3Kernel<<<grid, block, smem_bytes>>>(dA, dB, dC, M, N, K);
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));
            gpu_ms /= kRepeat;

            // 拷贝结果回主机
            CHECK_CUDA(cudaMemcpy(C_gpu.data(), dC, sz_c * sizeof(float), cudaMemcpyDeviceToHost));

            // 释放GPU资源
            CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
            CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC));
        }

        // 校验
        bool ok = true;
        double max_abs_diff = 0.0;
        const char* check = "SKIP_UNALIGNED";
        if (aligned && M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim) {
            ok = common::CheckEqual(C_cpu, C_gpu, 1e-3f);
            max_abs_diff = common::MaxAbsDiff(C_cpu, C_gpu);
            check = ok ? "PASS" : "FAIL";
        } else if (aligned) {
            check = "SKIP";
        }

        // 计算并输出结果
        double gflops = (gpu_ms > 0.0f) ? (2.0 * M * N * K / (gpu_ms * 1e6)) : 0.0;
        std::cout << M << "x" << N << "x" << K
                  << " | " << std::fixed << std::setprecision(4) << gpu_ms << " ms"
                  << " | " << std::setprecision(1) << gflops << " GFLOP/s"
                  << " | " << check << "\n";
        ofs << i << ",gemm_v3," << M << "," << N << "," << K << ","
            << gpu_ms << "," << gflops << "," << max_abs_diff << "," << check << "\n";
    }
    return 0;
}

#endif /* ALL_COMPARE_LIB */
