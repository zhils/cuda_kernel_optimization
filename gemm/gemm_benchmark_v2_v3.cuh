#pragma once
// Device code aligned with gemm_v2.cu / gemm_v3.cu (included by benchmark_all.cu only).
// If you edit GemmV2Kernel / GemmV3Kernel in those .cu files, update this header to match.
#include <cuda_runtime.h>

namespace gemm_bench_v2 {
// Tile config: each thread computes TM x TN sub-block
constexpr int kTM = 8;
constexpr int kTN = 8;

// Block dimensions in threads
constexpr int kBlockThreadsX = 16;
constexpr int kBlockThreadsY = 16;

// Block handles kBlockThreadsY * kTM rows x kBlockThreadsX * kTN cols of C
constexpr int kBlockM = kBlockThreadsY * kTM;  // 128
constexpr int kBlockN = kBlockThreadsX * kTN;  // 128

// Tile size along K dimension
constexpr int kTileK = 16;

__global__ void GemmV2Kernel(const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {

    // Shared memory tiles
    __shared__ float As[kBlockM][kTileK];
    __shared__ float Bs[kTileK][kBlockN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kBlockThreadsX + tx;

    // Starting row/col in C for this thread's sub-block
    const int row_start = blockIdx.y * kBlockM + ty * kTM;
    const int col_start = blockIdx.x * kBlockN + tx * kTN;

    // Register accumulators: TM x TN
    float sum[kTM][kTN] = {};

    const int num_k_tiles = (K + kTileK - 1) / kTileK;

    for (int t = 0; t < num_k_tiles; ++t) {
        const int k0 = t * kTileK;

        // Collaborative loading of A tile into shared memory using float4
        // Total A elements: kBlockM * kTileK = 128 * 16 = 2048
        // float4 count: 2048 / 4 = 512
        const int total_a_float4 = (kBlockM * kTileK) / 4;
        const int a_float4_per_thread = (total_a_float4 + (kBlockThreadsX * kBlockThreadsY) - 1) / (kBlockThreadsX * kBlockThreadsY);

        for (int l = 0; l < a_float4_per_thread; ++l) {
            int idx = tid * a_float4_per_thread + l;
            if (idx < total_a_float4) {
                int r = idx / (kTileK / 4);
                int k_offset = (idx % (kTileK / 4)) * 4;
                int g_r = blockIdx.y * kBlockM + r;
                int g_k = k0 + k_offset;

                if (g_r < M && g_k + 3 < K) {
                    float4 v = __ldg(reinterpret_cast<const float4*>(A + g_r * K + g_k));
                    As[r][k_offset + 0] = v.x;
                    As[r][k_offset + 1] = v.y;
                    As[r][k_offset + 2] = v.z;
                    As[r][k_offset + 3] = v.w;
                } else {
                    As[r][k_offset + 0] = (g_r < M && g_k + 0 < K) ? A[g_r * K + g_k + 0] : 0.0f;
                    As[r][k_offset + 1] = (g_r < M && g_k + 1 < K) ? A[g_r * K + g_k + 1] : 0.0f;
                    As[r][k_offset + 2] = (g_r < M && g_k + 2 < K) ? A[g_r * K + g_k + 2] : 0.0f;
                    As[r][k_offset + 3] = (g_r < M && g_k + 3 < K) ? A[g_r * K + g_k + 3] : 0.0f;
                }
            }
        }

        // Collaborative loading of B tile into shared memory using float4
        // Total B elements: kTileK * kBlockN = 16 * 128 = 2048
        // float4 count: 2048 / 4 = 512
        const int total_b_float4 = (kTileK * kBlockN) / 4;
        const int b_float4_per_thread = (total_b_float4 + (kBlockThreadsX * kBlockThreadsY) - 1) / (kBlockThreadsX * kBlockThreadsY);

        for (int l = 0; l < b_float4_per_thread; ++l) {
            int idx = tid * b_float4_per_thread + l;
            if (idx < total_b_float4) {
                int k_idx = idx / (kBlockN / 4);
                int c_offset = (idx % (kBlockN / 4)) * 4;
                int g_k = k0 + k_idx;
                int g_c = blockIdx.x * kBlockN + c_offset;

                if (g_k < K && g_c + 3 < N) {
                    float4 v = __ldg(reinterpret_cast<const float4*>(B + g_k * N + g_c));
                    Bs[k_idx][c_offset + 0] = v.x;
                    Bs[k_idx][c_offset + 1] = v.y;
                    Bs[k_idx][c_offset + 2] = v.z;
                    Bs[k_idx][c_offset + 3] = v.w;
                } else {
                    Bs[k_idx][c_offset + 0] = (g_k < K && g_c + 0 < N) ? B[g_k * N + g_c + 0] : 0.0f;
                    Bs[k_idx][c_offset + 1] = (g_k < K && g_c + 1 < N) ? B[g_k * N + g_c + 1] : 0.0f;
                    Bs[k_idx][c_offset + 2] = (g_k < K && g_c + 2 < N) ? B[g_k * N + g_c + 2] : 0.0f;
                    Bs[k_idx][c_offset + 3] = (g_k < K && g_c + 3 < N) ? B[g_k * N + g_c + 3] : 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute: each thread processes its TM x TN sub-block
        #pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            // Load B values for this thread's TN columns into registers
            float b_vals[kTN];
            #pragma unroll
            for (int j = 0; j < kTN; ++j) {
                b_vals[j] = Bs[kk][tx * kTN + j];
            }

            #pragma unroll
            for (int i = 0; i < kTM; ++i) {
                float a_val = As[ty * kTM + i][kk];
                #pragma unroll
                for (int j = 0; j < kTN; ++j) {
                    sum[i][j] += a_val * b_vals[j];
                }
            }
        }

        __syncthreads();
    }

    // Write results to C using float4 (8 elements = 2 float4 per row)
    #pragma unroll
    for (int i = 0; i < kTM; ++i) {
        int g_r = row_start + i;
        if (g_r >= M) continue;

        int g_c = col_start;
        // First float4 (columns 0-3)
        if (g_c + 3 < N) {
            float4 v;
            v.x = sum[i][0];
            v.y = sum[i][1];
            v.z = sum[i][2];
            v.w = sum[i][3];
            *reinterpret_cast<float4*>(C + g_r * N + g_c) = v;
        } else {
            for (int j = 0; j < 4 && g_c + j < N; ++j) {
                C[g_r * N + g_c + j] = sum[i][j];
            }
        }
        // Second float4 (columns 4-7)
        if (g_c + 7 < N) {
            float4 v;
            v.x = sum[i][4];
            v.y = sum[i][5];
            v.z = sum[i][6];
            v.w = sum[i][7];
            *reinterpret_cast<float4*>(C + g_r * N + g_c + 4) = v;
        } else if (g_c + 4 < N) {
            for (int j = 4; j < kTN && g_c + j < N; ++j) {
                C[g_r * N + g_c + j] = sum[i][j];
            }
        }
    }
}
}

namespace gemm_bench_v3 {
// Tile config: each thread computes TM x TN sub-block
constexpr int kTM = 8;
constexpr int kTN = 8;

// Block dimensions in threads
constexpr int kBlockThreadsX = 16;
constexpr int kBlockThreadsY = 16;

// Block handles kBlockThreadsY * kTM rows x kBlockThreadsX * kTN cols of C
constexpr int kBlockM = kBlockThreadsY * kTM;  // 128
constexpr int kBlockN = kBlockThreadsX * kTN;  // 128

// Tile size along K dimension
constexpr int kTileK = 16;

// Double buffer shared memory layout
// As[2][kBlockM][kTileK], Bs[2][kTileK][kBlockN]

__global__ void GemmV3Kernel(const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[2][kBlockM][kTileK];
    __shared__ float Bs[2][kTileK][kBlockN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kBlockThreadsX + tx;

    const int row_start = blockIdx.y * kBlockM + ty * kTM;
    const int col_start = blockIdx.x * kBlockN + tx * kTN;

    // Register accumulators
    float sum[kTM][kTN] = {};

    const int num_k_tiles = (K + kTileK - 1) / kTileK;

    // Preload first tile into buffer 0
    {
        const int k0 = 0;

        // Load A tile
        // Total A elements: kBlockM * kTileK = 128 * 16 = 2048
        // float4 count: 2048 / 4 = 512
        const int total_a_float4 = (kBlockM * kTileK) / 4;
        const int a_float4_per_thread = (total_a_float4 + (kBlockThreadsX * kBlockThreadsY) - 1) / (kBlockThreadsX * kBlockThreadsY);

        for (int l = 0; l < a_float4_per_thread; ++l) {
            int idx = tid * a_float4_per_thread + l;
            if (idx < total_a_float4) {
                int r = idx / (kTileK / 4);
                int k_offset = (idx % (kTileK / 4)) * 4;
                int g_r = blockIdx.y * kBlockM + r;
                int g_k = k0 + k_offset;

                if (g_r < M && g_k + 3 < K) {
                    float4 v = __ldg(reinterpret_cast<const float4*>(A + g_r * K + g_k));
                    As[0][r][k_offset + 0] = v.x;
                    As[0][r][k_offset + 1] = v.y;
                    As[0][r][k_offset + 2] = v.z;
                    As[0][r][k_offset + 3] = v.w;
                } else {
                    As[0][r][k_offset + 0] = (g_r < M && g_k + 0 < K) ? A[g_r * K + g_k + 0] : 0.0f;
                    As[0][r][k_offset + 1] = (g_r < M && g_k + 1 < K) ? A[g_r * K + g_k + 1] : 0.0f;
                    As[0][r][k_offset + 2] = (g_r < M && g_k + 2 < K) ? A[g_r * K + g_k + 2] : 0.0f;
                    As[0][r][k_offset + 3] = (g_r < M && g_k + 3 < K) ? A[g_r * K + g_k + 3] : 0.0f;
                }
            }
        }

        // Load B tile
        // Total B elements: kTileK * kBlockN = 16 * 128 = 2048
        // float4 count: 2048 / 4 = 512
        const int total_b_float4 = (kTileK * kBlockN) / 4;
        const int b_float4_per_thread = (total_b_float4 + (kBlockThreadsX * kBlockThreadsY) - 1) / (kBlockThreadsX * kBlockThreadsY);

        for (int l = 0; l < b_float4_per_thread; ++l) {
            int idx = tid * b_float4_per_thread + l;
            if (idx < total_b_float4) {
                int k_idx = idx / (kBlockN / 4);
                int c_offset = (idx % (kBlockN / 4)) * 4;
                int g_k = k0 + k_idx;
                int g_c = blockIdx.x * kBlockN + c_offset;

                if (g_k < K && g_c + 3 < N) {
                    float4 v = __ldg(reinterpret_cast<const float4*>(B + g_k * N + g_c));
                    Bs[0][k_idx][c_offset + 0] = v.x;
                    Bs[0][k_idx][c_offset + 1] = v.y;
                    Bs[0][k_idx][c_offset + 2] = v.z;
                    Bs[0][k_idx][c_offset + 3] = v.w;
                } else {
                    Bs[0][k_idx][c_offset + 0] = (g_k < K && g_c + 0 < N) ? B[g_k * N + g_c + 0] : 0.0f;
                    Bs[0][k_idx][c_offset + 1] = (g_k < K && g_c + 1 < N) ? B[g_k * N + g_c + 1] : 0.0f;
                    Bs[0][k_idx][c_offset + 2] = (g_k < K && g_c + 2 < N) ? B[g_k * N + g_c + 2] : 0.0f;
                    Bs[0][k_idx][c_offset + 3] = (g_k < K && g_c + 3 < N) ? B[g_k * N + g_c + 3] : 0.0f;
                }
            }
        }
    }
    __syncthreads();

    int read_buf = 0;
    int write_buf = 1;

    for (int t = 0; t < num_k_tiles; ++t) {
        const int k0 = t * kTileK;
        const int next_k0 = (t + 1) * kTileK;

        // Compute from read_buf
        #pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            float b_vals[kTN];
            #pragma unroll
            for (int j = 0; j < kTN; ++j) {
                b_vals[j] = Bs[read_buf][kk][tx * kTN + j];
            }

            #pragma unroll
            for (int i = 0; i < kTM; ++i) {
                float a_val = As[read_buf][ty * kTM + i][kk];
                #pragma unroll
                for (int j = 0; j < kTN; ++j) {
                    sum[i][j] += a_val * b_vals[j];
                }
            }
        }

        // If not last iteration, preload next tile into write_buf
        if (t + 1 < num_k_tiles) {
            // Load A tile into write_buf
            const int total_a_float4 = (kBlockM * kTileK) / 4;
            const int a_float4_per_thread = (total_a_float4 + (kBlockThreadsX * kBlockThreadsY) - 1) / (kBlockThreadsX * kBlockThreadsY);

            for (int l = 0; l < a_float4_per_thread; ++l) {
                int idx = tid * a_float4_per_thread + l;
                if (idx < total_a_float4) {
                    int r = idx / (kTileK / 4);
                    int k_offset = (idx % (kTileK / 4)) * 4;
                    int g_r = blockIdx.y * kBlockM + r;
                    int g_k = next_k0 + k_offset;

                    if (g_r < M && g_k + 3 < K) {
                        float4 v = __ldg(reinterpret_cast<const float4*>(A + g_r * K + g_k));
                        As[write_buf][r][k_offset + 0] = v.x;
                        As[write_buf][r][k_offset + 1] = v.y;
                        As[write_buf][r][k_offset + 2] = v.z;
                        As[write_buf][r][k_offset + 3] = v.w;
                    } else {
                        As[write_buf][r][k_offset + 0] = (g_r < M && g_k + 0 < K) ? A[g_r * K + g_k + 0] : 0.0f;
                        As[write_buf][r][k_offset + 1] = (g_r < M && g_k + 1 < K) ? A[g_r * K + g_k + 1] : 0.0f;
                        As[write_buf][r][k_offset + 2] = (g_r < M && g_k + 2 < K) ? A[g_r * K + g_k + 2] : 0.0f;
                        As[write_buf][r][k_offset + 3] = (g_r < M && g_k + 3 < K) ? A[g_r * K + g_k + 3] : 0.0f;
                    }
                }
            }

            // Load B tile into write_buf
            const int total_b_float4 = (kTileK * kBlockN) / 4;
            const int b_float4_per_thread = (total_b_float4 + (kBlockThreadsX * kBlockThreadsY) - 1) / (kBlockThreadsX * kBlockThreadsY);

            for (int l = 0; l < b_float4_per_thread; ++l) {
                int idx = tid * b_float4_per_thread + l;
                if (idx < total_b_float4) {
                    int k_idx = idx / (kBlockN / 4);
                    int c_offset = (idx % (kBlockN / 4)) * 4;
                    int g_k = next_k0 + k_idx;
                    int g_c = blockIdx.x * kBlockN + c_offset;

                    if (g_k < K && g_c + 3 < N) {
                        float4 v = __ldg(reinterpret_cast<const float4*>(B + g_k * N + g_c));
                        Bs[write_buf][k_idx][c_offset + 0] = v.x;
                        Bs[write_buf][k_idx][c_offset + 1] = v.y;
                        Bs[write_buf][k_idx][c_offset + 2] = v.z;
                        Bs[write_buf][k_idx][c_offset + 3] = v.w;
                    } else {
                        Bs[write_buf][k_idx][c_offset + 0] = (g_k < K && g_c + 0 < N) ? B[g_k * N + g_c + 0] : 0.0f;
                        Bs[write_buf][k_idx][c_offset + 1] = (g_k < K && g_c + 1 < N) ? B[g_k * N + g_c + 1] : 0.0f;
                        Bs[write_buf][k_idx][c_offset + 2] = (g_k < K && g_c + 2 < N) ? B[g_k * N + g_c + 2] : 0.0f;
                        Bs[write_buf][k_idx][c_offset + 3] = (g_k < K && g_c + 3 < N) ? B[g_k * N + g_c + 3] : 0.0f;
                    }
                }
            }
        }

        __syncthreads();

        // Swap buffers
        int tmp = read_buf;
        read_buf = write_buf;
        write_buf = tmp;
    }

    // Write results to C using float4 (8 elements = 2 float4 per row)
    #pragma unroll
    for (int i = 0; i < kTM; ++i) {
        int g_r = row_start + i;
        if (g_r >= M) continue;

        int g_c = col_start;
        // First float4 (columns 0-3)
        if (g_c + 3 < N) {
            float4 v;
            v.x = sum[i][0];
            v.y = sum[i][1];
            v.z = sum[i][2];
            v.w = sum[i][3];
            *reinterpret_cast<float4*>(C + g_r * N + g_c) = v;
        } else {
            for (int j = 0; j < 4 && g_c + j < N; ++j) {
                C[g_r * N + g_c + j] = sum[i][j];
            }
        }
        // Second float4 (columns 4-7)
        if (g_c + 7 < N) {
            float4 v;
            v.x = sum[i][4];
            v.y = sum[i][5];
            v.z = sum[i][6];
            v.w = sum[i][7];
            *reinterpret_cast<float4*>(C + g_r * N + g_c + 4) = v;
        } else if (g_c + 4 < N) {
            for (int j = 4; j < kTN && g_c + j < N; ++j) {
                C[g_r * N + g_c + j] = sum[i][j];
            }
        }
    }
}
}
