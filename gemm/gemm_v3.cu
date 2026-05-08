// GEMM V3: cp.async + 8×4 sub-block + TileK=32
//
// 关键参数调优：
//   1. kTN=4 (8×4): 累加器从 64 降到 32 寄存器 → 更高 occupancy
//   2. kBlockN=64 (128×64): B SMEM 减半 → 48KB → 2 blocks/SM
//   3. kTileK=32: 外循环从 256 次减到 128 次 → 更少 __syncthreads
//   4. cp.async: DMA 异步加载与计算重叠
//   5. __launch_bounds__(256,2): 预留 2 blocks/SM 的寄存器

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

constexpr int kTM = 8;
constexpr int kTN = 4;

constexpr int kBlockThreadsX = 16;
constexpr int kBlockThreadsY = 16;
constexpr int kBlockThreads = kBlockThreadsX * kBlockThreadsY;

constexpr int kBlockM = kBlockThreadsY * kTM;  // 128
constexpr int kBlockN = kBlockThreadsX * kTN;  // 64

constexpr int kTileK = 32;

constexpr int kSmemABuf = kBlockM * kTileK;  // 4096 floats
constexpr int kSmemBBuf = kTileK * kBlockN;  // 2048 floats

__global__ __launch_bounds__(kBlockThreads, 2)
void GemmV3Kernel(const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {

    extern __shared__ float smem[];
    float* As_buf0 = smem;
    float* As_buf1 = smem + kSmemABuf;
    float* Bs_buf0 = smem + 2 * kSmemABuf;
    float* Bs_buf1 = smem + 2 * kSmemABuf + kSmemBBuf;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kBlockThreadsX + tx;

    const int row_start = blockIdx.y * kBlockM + ty * kTM;
    const int col_start = blockIdx.x * kBlockN + tx * kTN;

    float sum[kTM][kTN] = {};

    const int num_k_tiles = (K + kTileK - 1) / kTileK;

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

    load_tile_async(0, As_buf0, Bs_buf0);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    for (int t = 0; t < num_k_tiles; ++t) {
        float* As_read = (t & 1) ? As_buf1 : As_buf0;
        float* Bs_read = (t & 1) ? Bs_buf1 : Bs_buf0;

        if (t + 1 < num_k_tiles) {
            float* As_write = (t & 1) ? As_buf0 : As_buf1;
            float* Bs_write = (t & 1) ? Bs_buf0 : Bs_buf1;
            load_tile_async((t + 1) * kTileK, As_write, Bs_write);
            __pipeline_commit();
        }

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

static void GemmCPU(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) {
            float s = 0;
            for (int k = 0; k < K; ++k)
                s += A[static_cast<size_t>(r) * K + k] * B[static_cast<size_t>(k) * N + c];
            C[static_cast<size_t>(r) * N + c] = s;
        }
}

int main() {
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

        std::vector<float> A(sz_a), B(sz_b), C_cpu(sz_c), C_gpu(sz_c);
        common::InitMatrix(A, M, K);
        common::InitMatrix(B, K, N);
        if (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim)
            GemmCPU(A.data(), B.data(), C_cpu.data(), M, N, K);

        float gpu_ms = 0.0f;
        if (aligned) {
            float *dA, *dB, *dC;
            CHECK_CUDA(cudaMalloc(&dA, sz_a * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&dB, sz_b * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&dC, sz_c * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(dA, A.data(), sz_a * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(dB, B.data(), sz_b * sizeof(float), cudaMemcpyHostToDevice));

            dim3 block(kBlockThreadsX, kBlockThreadsY);
            dim3 grid((N + kBlockN - 1) / kBlockN, (M + kBlockM - 1) / kBlockM);

            CHECK_CUDA(cudaFuncSetAttribute(GemmV3Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

            GemmV3Kernel<<<grid, block, smem_bytes>>>(dA, dB, dC, M, N, K);
            CHECK_CUDA(cudaDeviceSynchronize());

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

            CHECK_CUDA(cudaMemcpy(C_gpu.data(), dC, sz_c * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
            CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC));
        }

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
