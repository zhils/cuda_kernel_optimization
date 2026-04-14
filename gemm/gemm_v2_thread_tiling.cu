// GEMM V2: Thread-level tiling — each thread computes a TM×TN sub-block
//
// Key optimizations over V1:
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

__global__ void GemmV2Kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {
    constexpr int TILE_K = 8;
    constexpr int TM = 4;
    constexpr int TN = 4;
    constexpr int BLOCK_ROWS = 32;
    constexpr int BLOCK_COLS = 32;

    // Single-buffered transposed + padded shared memory layout.
    __shared__ float As[BLOCK_ROWS][TILE_K + 1];
    __shared__ float Bs[BLOCK_COLS][TILE_K + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_start = blockIdx.y * BLOCK_ROWS + ty * TM;
    int col_start = blockIdx.x * BLOCK_COLS + tx * TN;

    // v2_opt1: scalar accumulators + instruction reordering to relax RAW dependency chain.
    float sum00 = 0.0f, sum01 = 0.0f, sum02 = 0.0f, sum03 = 0.0f;
    float sum10 = 0.0f, sum11 = 0.0f, sum12 = 0.0f, sum13 = 0.0f;
    float sum20 = 0.0f, sum21 = 0.0f, sum22 = 0.0f, sum23 = 0.0f;
    float sum30 = 0.0f, sum31 = 0.0f, sum32 = 0.0f, sum33 = 0.0f;

    const int tiles = (K + TILE_K - 1) / TILE_K;
    const int tid = ty * blockDim.x + tx;
    auto load_tile = [&](int tile_idx) {
        const int k_start = tile_idx * TILE_K;
        for (int idx = tid; idx < TILE_K * BLOCK_ROWS; idx += blockDim.x * blockDim.y) {
            const int k_idx = idx / BLOCK_ROWS;
            const int row_idx = idx % BLOCK_ROWS;
            const int global_row = blockIdx.y * BLOCK_ROWS + row_idx;
            const int global_col = k_start + k_idx;
            As[row_idx][k_idx] = (global_row < M && global_col < K)
                                     ? __ldg(A + global_row * K + global_col)
                                     : 0.0f;
        }

        for (int idx = tid; idx < TILE_K * BLOCK_COLS; idx += blockDim.x * blockDim.y) {
            const int k_idx = idx / BLOCK_COLS;
            const int col_idx = idx % BLOCK_COLS;
            const int global_row = k_start + k_idx;
            const int global_col = blockIdx.x * BLOCK_COLS + col_idx;
            Bs[col_idx][k_idx] = (global_row < K && global_col < N)
                                     ? __ldg(B + global_row * N + global_col)
                                     : 0.0f;
        }
    };

    for (int t = 0; t < tiles; ++t) {
        load_tile(t);
        __syncthreads();

        #pragma unroll 4
        for (int k = 0; k < TILE_K; ++k) {
            const float b0 = Bs[tx * TN + 0][k];
            const float b1 = Bs[tx * TN + 1][k];
            const float b2 = Bs[tx * TN + 2][k];
            const float b3 = Bs[tx * TN + 3][k];

            // Balanced mode: short live range + partial interleaving.
            float a = As[ty * TM + 0][k];
            sum00 += a * b0; sum01 += a * b1;
            sum02 += a * b2; sum03 += a * b3;

            a = As[ty * TM + 1][k];
            sum11 += a * b1; sum10 += a * b0;
            sum13 += a * b3; sum12 += a * b2;

            a = As[ty * TM + 2][k];
            sum22 += a * b2; sum23 += a * b3;
            sum20 += a * b0; sum21 += a * b1;

            a = As[ty * TM + 3][k];
            sum33 += a * b3; sum32 += a * b2;
            sum31 += a * b1; sum30 += a * b0;
        }
        __syncthreads();
    }

    const int r0 = row_start + 0, r1 = row_start + 1, r2 = row_start + 2, r3 = row_start + 3;
    const int c0 = col_start + 0, c1 = col_start + 1, c2 = col_start + 2, c3 = col_start + 3;
    if (r0 < M && c0 < N) C[r0 * N + c0] = sum00;
    if (r0 < M && c1 < N) C[r0 * N + c1] = sum01;
    if (r0 < M && c2 < N) C[r0 * N + c2] = sum02;
    if (r0 < M && c3 < N) C[r0 * N + c3] = sum03;
    if (r1 < M && c0 < N) C[r1 * N + c0] = sum10;
    if (r1 < M && c1 < N) C[r1 * N + c1] = sum11;
    if (r1 < M && c2 < N) C[r1 * N + c2] = sum12;
    if (r1 < M && c3 < N) C[r1 * N + c3] = sum13;
    if (r2 < M && c0 < N) C[r2 * N + c0] = sum20;
    if (r2 < M && c1 < N) C[r2 * N + c1] = sum21;
    if (r2 < M && c2 < N) C[r2 * N + c2] = sum22;
    if (r2 < M && c3 < N) C[r2 * N + c3] = sum23;
    if (r3 < M && c0 < N) C[r3 * N + c0] = sum30;
    if (r3 < M && c1 < N) C[r3 * N + c1] = sum31;
    if (r3 < M && c2 < N) C[r3 * N + c2] = sum32;
    if (r3 < M && c3 < N) C[r3 * N + c3] = sum33;
}

// Fast path for aligned/full tiles: no boundary predicates in main loop + vectorized global loads.
__global__ void GemmV2KernelFast(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
    constexpr int TILE_K = 8;
    constexpr int TM = 4;
    constexpr int TN = 4;
    constexpr int BLOCK_ROWS = 32;
    constexpr int BLOCK_COLS = 32;

    __shared__ float As[BLOCK_ROWS][TILE_K + 1];
    __shared__ float Bs[BLOCK_COLS][TILE_K + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int row_start = blockIdx.y * BLOCK_ROWS + ty * TM;
    const int col_start = blockIdx.x * BLOCK_COLS + tx * TN;

    float sum00 = 0.0f, sum01 = 0.0f, sum02 = 0.0f, sum03 = 0.0f;
    float sum10 = 0.0f, sum11 = 0.0f, sum12 = 0.0f, sum13 = 0.0f;
    float sum20 = 0.0f, sum21 = 0.0f, sum22 = 0.0f, sum23 = 0.0f;
    float sum30 = 0.0f, sum31 = 0.0f, sum32 = 0.0f, sum33 = 0.0f;

    const int tiles = K / TILE_K;
    for (int t = 0; t < tiles; ++t) {
        const int k_start = t * TILE_K;

        // A tile: vectorized load along K dimension (2 float4 per row).
        if (tid < BLOCK_ROWS * (TILE_K / 4)) {
            const int row_idx = tid / (TILE_K / 4);
            const int k4 = tid % (TILE_K / 4);
            const int global_row = blockIdx.y * BLOCK_ROWS + row_idx;
            const float4 v = reinterpret_cast<const float4*>(
                                 A + global_row * K + k_start)[k4];
            As[row_idx][k4 * 4 + 0] = v.x;
            As[row_idx][k4 * 4 + 1] = v.y;
            As[row_idx][k4 * 4 + 2] = v.z;
            As[row_idx][k4 * 4 + 3] = v.w;
        }

        // B tile: vectorized load along N dimension (8 rows * 8 float4).
        if (tid < TILE_K * (BLOCK_COLS / 4)) {
            const int k_idx = tid / (BLOCK_COLS / 4);
            const int c4 = tid % (BLOCK_COLS / 4);
            const int global_row = k_start + k_idx;
            const int global_col = blockIdx.x * BLOCK_COLS + c4 * 4;
            const float4 v = reinterpret_cast<const float4*>(
                                 B + global_row * N + global_col)[0];
            Bs[c4 * 4 + 0][k_idx] = v.x;
            Bs[c4 * 4 + 1][k_idx] = v.y;
            Bs[c4 * 4 + 2][k_idx] = v.z;
            Bs[c4 * 4 + 3][k_idx] = v.w;
        }

        __syncthreads();

        #pragma unroll 4
        for (int k = 0; k < TILE_K; ++k) {
            const float b0 = Bs[tx * TN + 0][k];
            const float b1 = Bs[tx * TN + 1][k];
            const float b2 = Bs[tx * TN + 2][k];
            const float b3 = Bs[tx * TN + 3][k];

            float a = As[ty * TM + 0][k];
            sum00 += a * b0; sum01 += a * b1;
            sum02 += a * b2; sum03 += a * b3;

            a = As[ty * TM + 1][k];
            sum11 += a * b1; sum10 += a * b0;
            sum13 += a * b3; sum12 += a * b2;

            a = As[ty * TM + 2][k];
            sum22 += a * b2; sum23 += a * b3;
            sum20 += a * b0; sum21 += a * b1;

            a = As[ty * TM + 3][k];
            sum33 += a * b3; sum32 += a * b2;
            sum31 += a * b1; sum30 += a * b0;
        }
        __syncthreads();
    }

    const int r0 = row_start + 0, r1 = row_start + 1, r2 = row_start + 2, r3 = row_start + 3;
    const int c0 = col_start + 0, c1 = col_start + 1, c2 = col_start + 2, c3 = col_start + 3;
    C[r0 * N + c0] = sum00; C[r0 * N + c1] = sum01; C[r0 * N + c2] = sum02; C[r0 * N + c3] = sum03;
    C[r1 * N + c0] = sum10; C[r1 * N + c1] = sum11; C[r1 * N + c2] = sum12; C[r1 * N + c3] = sum13;
    C[r2 * N + c0] = sum20; C[r2 * N + c1] = sum21; C[r2 * N + c2] = sum22; C[r2 * N + c3] = sum23;
    C[r3 * N + c0] = sum30; C[r3 * N + c1] = sum31; C[r3 * N + c2] = sum32; C[r3 * N + c3] = sum33;
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
    constexpr int kMaxGpuRunDim = 4096;
    constexpr int kMaxCpuVerifyDim = 1024;
    constexpr int kTileRows = 32;
    constexpr int kTileCols = 32;
    auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/gemm_v2_results.csv");
    ofs << "id,M,N,K,gpu_ms,gflops,max_abs_diff,check\n";

    for (size_t i = 0; i < cases.size(); ++i) {
        int M = cases[i].rows, N = cases[i].cols, K = (M + N) / 2;
        const bool do_gpu_run = (M <= kMaxGpuRunDim && N <= kMaxGpuRunDim && K <= kMaxGpuRunDim);
        std::vector<float> A(static_cast<size_t>(M) * K),
            B(static_cast<size_t>(K) * N),
            cpu(static_cast<size_t>(M) * N),
            gpu(static_cast<size_t>(M) * N);
        common::InitMatrix(A, M, K);
        common::InitMatrix(B, K, N);
        const bool do_cpu_verify = (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim && K <= kMaxCpuVerifyDim);
        if (do_cpu_verify) {
            GemmCPU(A.data(), B.data(), cpu.data(), M, N, K);
        }

        float gpu_ms = 0.0f;
        if (do_gpu_run) {
            float *dA, *dB, *dC;
            CHECK_CUDA(cudaMalloc(&dA, A.size() * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&dB, B.size() * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&dC, cpu.size() * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

            dim3 block(8, 8);
            dim3 grid((N + kTileCols - 1) / kTileCols, (M + kTileRows - 1) / kTileRows);

            const bool fast_path = ((M % 32) == 0 && (N % 32) == 0 && (K % 8) == 0);
            if (fast_path) {
                GemmV2KernelFast<<<grid, block>>>(dA, dB, dC, M, N, K);
            } else {
                GemmV2Kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
            }
            CHECK_CUDA(cudaDeviceSynchronize());

            cudaEvent_t s, e;
            CHECK_CUDA(cudaEventCreate(&s));
            CHECK_CUDA(cudaEventCreate(&e));
            std::vector<float> gpu_times;
            gpu_times.reserve(kRepeat);
            for (int rep = 0; rep < kRepeat; ++rep) {
                CHECK_CUDA(cudaEventRecord(s));
                if (fast_path) {
                    GemmV2KernelFast<<<grid, block>>>(dA, dB, dC, M, N, K);
                } else {
                    GemmV2Kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
                }
                CHECK_CUDA(cudaEventRecord(e));
                CHECK_CUDA(cudaEventSynchronize(e));
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
        } else if (do_cpu_verify) {
            ok = common::CheckEqual(cpu, gpu, 1e-3f);
            max_abs_diff = common::MaxAbsDiff(cpu, gpu);
            check = ok ? "PASS" : "FAIL";
        }
        double gflops = (gpu_ms > 0.0f) ? (2.0 * M * N * K / (gpu_ms * 1e6)) : 0.0;

        std::cout << "M=" << M << " N=" << N << " K=" << K
                  << " | " << std::fixed << std::setprecision(3) << gpu_ms << " ms"
                  << " | " << std::setprecision(1) << gflops << " GFLOPS"
                  << " | " << check << "\n";

        ofs << i << "," << M << "," << N << "," << K << ","
            << gpu_ms << "," << gflops << ","
            << max_abs_diff << "," << check << "\n";

    }
    return 0;
}
