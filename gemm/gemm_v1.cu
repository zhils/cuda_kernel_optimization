// GEMM V1: classic tiling approach with 16x16 tile
//
// Design:
// - Each block handles a 16x16 tile of C
// - Load tile from A and B into shared memory
// - Each thread computes 1 element of C

#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

__global__ void GemmV1Kernel(const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {

    // Block position in C
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;

    // Thread position within tile
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;

    // Global position in C
    const int row = block_row + thread_row;
    const int col = block_col + thread_col;

    if (row >= M || col >= N) return;

    // Shared memory for A tile and B tile
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    float sum = 0.0f;

    // Loop over K dimension in tiles
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_k_tiles; ++t) {
        const int k_start = t * TILE_K;

        // Load A tile: As[thread_row][thread_col] = A[row][k_start + thread_col]
        if (thread_col < TILE_K && k_start + thread_col < K) {
            As[thread_row][thread_col] = A[row * K + (k_start + thread_col)];
        } else {
            As[thread_row][thread_col] = 0.0f;
        }

        // Load B tile: Bs[thread_row][thread_col] = B[k_start + thread_row][col]
        if (thread_row < TILE_K && k_start + thread_row < K) {
            Bs[thread_row][thread_col] = B[(k_start + thread_row) * N + col];
        } else {
            Bs[thread_row][thread_col] = 0.0f;
        }
        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += As[thread_row][k] * Bs[k][thread_col];
        }
        __syncthreads();
    }

    // Write result to C
    C[row * N + col] = sum;
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
    constexpr int kMaxCpuVerifyDim = 512;
    auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/gemm_v1_results.csv");
    ofs << "id,group,M,N,K,gpu_ms,gflops,max_abs_diff,check\n";

    for (size_t i = 0; i < cases.size(); ++i) {
        int M = cases[i].rows, N = cases[i].cols, K = M;
        std::vector<float> A(static_cast<size_t>(M) * K),
                           B(static_cast<size_t>(K) * N),
                           C_cpu(static_cast<size_t>(M) * N),
                           C_gpu(static_cast<size_t>(M) * N);
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

        dim3 block(TILE_N, TILE_M);
        dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

        // Warmup
        GemmV1Kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        for (int r = 0; r < kRepeat; ++r) {
            GemmV1Kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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

        ofs << i << ",gemm_v1," << M << "," << N << "," << K << ","
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
