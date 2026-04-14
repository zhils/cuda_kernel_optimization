// GEMM V3: Thread-level tiling — each thread computes a TM×TN sub-block
//
// Key optimizations over V2:
//   - Each thread computes 4×4 output elements (register tiling)
//   - Reduced thread count per block → higher register budget per thread
//   - Improved arithmetic intensity by reusing loaded values in registers

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

__global__ void GemmV3Kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {
    constexpr int TILE_K = 8;
    constexpr int TM = 4;
    constexpr int TN = 4;
    constexpr int BLOCK_ROWS = 32;
    constexpr int BLOCK_COLS = 32;

    __shared__ float As[TILE_K][BLOCK_ROWS];
    __shared__ float Bs[TILE_K][BLOCK_COLS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_start = blockIdx.y * BLOCK_ROWS + ty * TM;
    int col_start = blockIdx.x * BLOCK_COLS + tx * TN;

    float sum[TM][TN] = {{0.0f}};

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {
        int k_start = t * TILE_K;

        int load_row = ty * 4 + tx / 8;
        int load_col = tx % 8;

        if (load_row < BLOCK_ROWS && load_col < TILE_K) {
            int global_row = blockIdx.y * BLOCK_ROWS + load_row;
            int global_col = k_start + load_col;
            As[load_col][load_row] = (global_row < M && global_col < K)
                                         ? __ldg(A + global_row * K + global_col)
                                         : 0.0f;
        }

        int load_row_b = tx / 8;
        int load_col_b = ty * 4 + tx % 4;

        if (load_row_b < TILE_K && load_col_b < BLOCK_COLS) {
            int global_row_b = k_start + load_row_b;
            int global_col_b = blockIdx.x * BLOCK_COLS + load_col_b;
            Bs[load_row_b][load_col_b] = (global_row_b < K && global_col_b < N)
                                              ? __ldg(B + global_row_b * N + global_col_b)
                                              : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            for (int i = 0; i < TM; ++i) {
                float a_val = As[k][ty * TM + i];
                for (int j = 0; j < TN; ++j) {
                    sum[i][j] += a_val * Bs[k][tx * TN + j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int r = row_start + i;
            int c = col_start + j;
            if (r < M && c < N) {
                C[r * N + c] = sum[i][j];
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
    auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/gemm_v3_results.csv");
    ofs << "id,M,N,K,gpu_ms,gflops,max_abs_diff,check\n";

    for (size_t i = 0; i < cases.size(); ++i) {
        int M = cases[i].rows, N = cases[i].cols, K = (M + N) / 2;
        std::vector<float> A(static_cast<size_t>(M) * K),
            B(static_cast<size_t>(K) * N),
            cpu(static_cast<size_t>(M) * N),
            gpu(static_cast<size_t>(M) * N);
        common::InitMatrix(A, M, K);
        common::InitMatrix(B, K, N);
        GemmCPU(A.data(), B.data(), cpu.data(), M, N, K);

        float *dA, *dB, *dC;
        CHECK_CUDA(cudaMalloc(&dA, A.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dB, B.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dC, cpu.size() * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(8, 8);
        dim3 grid((N + 31) / 32, (M + 31) / 32);

        GemmV3Kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        CHECK_CUDA(cudaEventRecord(s));
        for (int rep = 0; rep < kRepeat; ++rep)
            GemmV3Kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));
        float gpu_ms_total = 0;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_total, s, e));
        float gpu_ms = gpu_ms_total / kRepeat;

        CHECK_CUDA(cudaMemcpy(gpu.data(), dC, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = common::CheckEqual(cpu, gpu, 1e-3f);
        double gflops = 2.0 * M * N * K / (gpu_ms * 1e6);

        std::cout << "M=" << M << " N=" << N << " K=" << K
                  << " | " << std::fixed << std::setprecision(3) << gpu_ms << " ms"
                  << " | " << std::setprecision(1) << gflops << " GFLOPS"
                  << " | " << (ok ? "PASS" : "FAIL") << "\n";

        ofs << i << "," << M << "," << N << "," << K << ","
            << gpu_ms << "," << gflops << ","
            << common::MaxAbsDiff(cpu, gpu) << "," << (ok ? "PASS" : "FAIL") << "\n";

        CHECK_CUDA(cudaEventDestroy(s));
        CHECK_CUDA(cudaEventDestroy(e));
        CHECK_CUDA(cudaFree(dA));
        CHECK_CUDA(cudaFree(dB));
        CHECK_CUDA(cudaFree(dC));
    }
    return 0;
}
