// GEMM V2: thread-level register tiling with vectorized loads
//
// Design:
// - Block size: 16x16 threads (256 threads)
// - Each thread computes a 8x8 sub-block of C (TM=8, TN=8)
// - Block handles 128x128 of C (16*8 x 16*8)
// - Tile K = 16, loop over K in tiles
// - Use float4 to load A and B tiles into shared memory
// - Each thread keeps 8x8 = 64 accumulators in registers
// - Arithmetic intensity: 64 FMA / (8+8) shared mem loads = 4 FMA per load

#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

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
    auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/gemm_v2_results.csv");
    ofs << "id,group,M,N,K,gpu_ms,gflops,max_abs_diff,check\n";

    for (size_t i = 0; i < cases.size(); ++i) {
        int M = cases[i].rows, N = cases[i].cols, K = M;
        int n = M * N;
        std::vector<float> A(n), B(n), C_cpu(n), C_gpu(n);
        common::InitMatrix(A, M, K);
        common::InitMatrix(B, K, N);
        if (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim) {
            GemmCPU(A.data(), B.data(), C_cpu.data(), M, N, K);
        }

        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(kBlockThreadsX, kBlockThreadsY);
        dim3 grid((N + kBlockN - 1) / kBlockN, (M + kBlockM - 1) / kBlockM);

        // Warmup
        GemmV2Kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        for (int r = 0; r < kRepeat; ++r) {
            GemmV2Kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        ms /= kRepeat;

        CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = true;
        if (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim) {
            ok = common::CheckEqual(C_cpu, C_gpu, 1e-3f);
        }
        double gflops = (2.0 * M * N * K) / (ms * 1e6);

        std::cout << M << "x" << N << "x" << K << " | " << std::fixed << std::setprecision(4) << ms << " ms"
                  << " | " << std::setprecision(1) << gflops << " GFLOP/s"
                  << " | " << (ok ? "PASS" : "FAIL") << "\n";

        ofs << i << ",gemm_v2," << M << "," << N << "," << K << ","
            << ms << "," << gflops << ","
            << common::MaxAbsDiff(C_cpu, C_gpu) << ","
            << (ok ? "PASS" : "FAIL") << "\n";

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
    }
    return 0;
}
