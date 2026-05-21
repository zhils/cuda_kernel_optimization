#pragma once

#include <cuda_pipeline.h>
#include <cuda_runtime.h>

namespace gemm_v3 {

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

}  // namespace gemm_v3
