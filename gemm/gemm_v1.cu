#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kTileK = 16;

__global__ void GemmV1Kernel(const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {
    const int block_row = blockIdx.y * kTileM;
    const int block_col = blockIdx.x * kTileN;
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    const int row = block_row + thread_row;
    const int col = block_col + thread_col;

    if (row >= M || col >= N) return;

    __shared__ float As[kTileM][kTileK];
    __shared__ float Bs[kTileK][kTileN];

    float sum = 0.0f;

    const int num_k_tiles = (K + kTileK - 1) / kTileK;

    for (int t = 0; t < num_k_tiles; ++t) {
        const int k_start = t * kTileK;

        if (thread_col < kTileK && k_start + thread_col < K) {
            As[thread_row][thread_col] = A[row * K + (k_start + thread_col)];
        } else {
            As[thread_row][thread_col] = 0.0f;
        }

        if (thread_row < kTileK && k_start + thread_row < K) {
            Bs[thread_row][thread_col] = B[(k_start + thread_row) * N + col];
        } else {
            Bs[thread_row][thread_col] = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kTileK; ++k) {
            sum += As[thread_row][k] * Bs[k][thread_col];
        }
        __syncthreads();
    }

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

        dim3 block(kTileN, kTileM);
        dim3 grid((N + kTileN - 1) / kTileN, (M + kTileM - 1) / kTileM);

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
