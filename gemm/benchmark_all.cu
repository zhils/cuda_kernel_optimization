// GEMM Performance Comparison: Naive vs Optimized vs cuBLAS vs CUTLASS
// NVIDIA Library Implementations: cuBLAS, CUTLASS

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

#define CHECK_CUBLAS(call)                                                       \
  do {                                                                           \
    cublasStatus_t s__ = (call);                                                 \
    if (s__ != CUBLAS_STATUS_SUCCESS) {                                          \
      std::cerr << "cuBLAS error: " << static_cast<int>(s__) << std::endl;       \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

#include <mma.h>
using namespace nvcuda::wmma;

constexpr int BLOCK_SIZE = 256;
constexpr int TILE_K = 8;

// ============ Kernel 1: Naive GEMM ============
__global__ void GemmNaiveKernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============ Kernel 2: Shared Memory Tiling ============
__global__ void GemmSmemKernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
    constexpr int TILE = 16;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.0f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.0f;

        __syncthreads();

        for (int kk = 0; kk < TILE; ++kk) {
            sum += As[ty][kk] * Bs[kk][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============ Kernel 3: Optimized with __ldg and Bank Conflict Avoidance ============
__global__ void GemmOptimizedKernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K) {
    constexpr int TILE = 16;

    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        As[ty][tx] = (row < M && k0 + tx < K) ? __ldg(A + row * K + k0 + tx) : 0.0f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? __ldg(B + (k0 + ty) * N + col) : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            sum += As[ty][kk] * Bs[kk][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============ Kernel 4: Tensor Core (WMMA) - CUTLASS-style ============
__global__ void GemmTensorCoreKernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K) {
    int warpid = (threadIdx.x / 32);
    int laneid = (threadIdx.x % 32);

    int row_start = (blockIdx.y * 16 * 4) + warpid * 16;
    int col_start = (blockIdx.x * 16 * 4) + laneid / 4 * 4;
    int row = row_start + (laneid % 4);

    if (row >= M || col_start + 3 >= N) return;

    fragment<matrix_a, 16, 16, 16> fragA;
    fragment<matrix_b, 16, 16, 16> fragB;
    fragment<accumulator, 16, 16, 16> fragC;

    fill_fragment(fragC, 0.0f);

    int k_iterations = K / 16;

    for (int k = 0; k < k_iterations; ++k) {
        if (row < M && k * 16 + 15 < K) {
            load_matrix_sync(fragA, A + row * K + k * 16, K);
        }

        if (col_start + 3 < N && k * 16 + 15 < K) {
            load_matrix_sync(fragB, B + k * 16 * N + col_start, N);
        }

        mma_sync(fragC, fragA, fragB, fragC);
    }

    store_matrix_sync(C + row * N + col_start, fragC, N, mem_row_major);
}

// ============ cuBLAS Implementation ============
static double RunCuBLAS(cublasHandle_t handle,
                         const float* h_A, const float* h_B, float* h_C,
                         int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(s));

    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));

    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));

    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return ms;
}

// ============ CUTLASS-style GEMM (简化实现) ============
__global__ void GemmCutlassStyleKernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int N, int K) {
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 32;
    constexpr int THREADS = 256;
    constexpr int ELEMENTS_PER_THREAD = (TILE_M * TILE_N) / THREADS;

    int tid = threadIdx.x;
    int block_m = blockIdx.x;
    int block_n = blockIdx.y;

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int tile_m_start = block_m * TILE_M;
    int tile_n_start = block_n * TILE_N;

    float thread_sum[ELEMENTS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        thread_sum[i] = 0.0f;
    }

    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        __shared__ float As[TILE_M][TILE_K];
        __shared__ float Bs[TILE_K][TILE_N];

        int a_row_start = tile_m_start;
        int a_col = k_tile;
        int b_row = k_tile;
        int b_col_start = tile_n_start;

        if (warp_id < 8 && lane_id < 32) {
            int a_tile_row = warp_id / 2;
            int a_tile_col = warp_id % 2;
            int a_idx = a_tile_row * 16 + lane_id / 2;
            int a_jdx = a_tile_col * 16 + lane_id % 2;

            if (a_idx < TILE_M && a_jdx < TILE_K && a_row_start + a_idx < M && a_col + a_jdx < K) {
                As[a_idx][a_jdx] = __ldg(A + (a_row_start + a_idx) * K + a_col + a_jdx);
            }
        }

        if (warp_id < 8 && lane_id < 32) {
            int b_tile_row = warp_id / 4;
            int b_tile_col = warp_id % 4;
            int b_idx = b_tile_row * 8 + lane_id / 4;
            int b_jdx = b_tile_col * 16 + lane_id % 4;

            if (b_idx < TILE_K && b_jdx < TILE_N && b_row + b_idx < K && b_col_start + b_jdx < N) {
                Bs[b_idx][b_jdx] = __ldg(B + (b_row + b_idx) * N + b_col_start + b_jdx);
            }
        }

        __syncthreads();

        for (int kk = 0; kk < TILE_K; ++kk) {
            int c_row = tid / 16;
            int c_col = (tid % 16) * 8;

            for (int e = 0; e < ELEMENTS_PER_THREAD / 8; ++e) {
                int row = tile_m_start + c_row;
                int col = tile_n_start + c_col;

                if (row < M && col < N) {
                    thread_sum[e * 8] += As[c_row][kk] * Bs[kk][c_col];
                }

                c_col += 64;
                if (c_col >= TILE_N) {
                    c_col = (tid % 16) * 8;
                    c_row += 64;
                }
            }
        }

        __syncthreads();
    }

    for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
        int row = tile_m_start + (e / (TILE_N / 8)) * (THREADS / 16) + tid / 16;
        int col = tile_n_start + (e % (TILE_N / 8)) * 8 + (tid % 16);

        if (row < M && col < N) {
            C[row * N + col] = thread_sum[e];
        }
    }
}

// ============ CPU Reference ============
static void GemmCPU(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[r * K + k] * B[k * N + c];
            }
            C[r * N + c] = sum;
        }
    }
}

// ============ Benchmark Helpers ============
template<typename KernelFunc>
double RunKernel(KernelFunc kernel,
                  const float* h_A, const float* h_B, float* h_C,
                  int M, int N, int K,
                  dim3 grid, dim3 block, int iterations) {
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < 3; ++i) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<double> times;
    for (int iter = 0; iter < iterations; ++iter) {
        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));

        CHECK_CUDA(cudaEventRecord(s));
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
        times.push_back(ms);

        CHECK_CUDA(cudaEventDestroy(s));
        CHECK_CUDA(cudaEventDestroy(e));
    }

    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    std::sort(times.begin(), times.end());
    double sum = 0;
    for (size_t i = 1; i < times.size() - 1; ++i) {
        sum += times[i];
    }
    return sum / (times.size() - 2);
}

double ComputeGFLOPS(int M, int N, int K, double ms) {
    if (ms <= 0) return 0;
    double flops = 2.0 * M * N * K;
    return flops / (ms * 1e6);
}

// ============ Main ============
int main() {
    constexpr int ITERATIONS = 10;

    std::cout << "========================================\n";
    std::cout << "  GEMM Performance Comparison\n";
    std::cout << "  Naive | Smem | Optimized | TensorCore | cuBLAS | CUTLASS-style\n";
    std::cout << "========================================\n\n";

    std::vector<std::tuple<int, int, int>> test_cases = {
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {1024, 2048, 1024},
        {2048, 1024, 2048},
    };

    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/gemm_all_comparison.csv");
    ofs << "M,N,K,naive_ms,smem_ms,opt_ms,tensorcore_ms,cublas_ms,cutlass_ms";
    ofs << ",naive_gflops,smem_gflops,opt_gflops,tensorcore_gflops,cublas_gflops,cutlass_gflops\n";

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    std::cout << std::left
              << std::setw(6) << "M"
              << std::setw(6) << "N"
              << std::setw(6) << "K"
              << std::setw(12) << "Naive"
              << std::setw(12) << "Smem"
              << std::setw(12) << "Optimized"
              << std::setw(12) << "TensorCore"
              << std::setw(12) << "cuBLAS"
              << std::setw(12) << "CUTLASS"
              << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (const auto& [M, N, K] : test_cases) {
        std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_cpu(M * N);

        common::InitMatrix(h_A, M, K);
        common::InitMatrix(h_B, K, N);

        auto t0 = std::chrono::high_resolution_clock::now();
        GemmCPU(h_A.data(), h_B.data(), h_cpu.data(), M, N, K);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << std::left
                  << std::setw(6) << M
                  << std::setw(6) << N
                  << std::setw(6) << K;

        dim3 block_naive(16, 16);
        dim3 grid_naive((N + 15) / 16, (M + 15) / 16);
        double naive_ms = RunKernel(GemmNaiveKernel, h_A.data(), h_B.data(), h_C.data(),
                                     M, N, K, grid_naive, block_naive, ITERATIONS);
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(12) << naive_ms;

        dim3 block_smem(16, 16);
        dim3 grid_smem((N + 15) / 16, (M + 15) / 16);
        double smem_ms = RunKernel(GemmSmemKernel, h_A.data(), h_B.data(), h_C.data(),
                                    M, N, K, grid_smem, block_smem, ITERATIONS);
        std::cout << std::setw(12) << smem_ms;

        dim3 block_opt(16, 16);
        dim3 grid_opt((N + 15) / 16, (M + 15) / 16);
        double opt_ms = RunKernel(GemmOptimizedKernel, h_A.data(), h_B.data(), h_C.data(),
                                   M, N, K, grid_opt, block_opt, ITERATIONS);
        std::cout << std::setw(12) << opt_ms;

        dim3 block_tc(256, 1);
        dim3 grid_tc((N + 63) / 64, (M + 63) / 64);
        double tensorcore_ms = RunKernel(GemmTensorCoreKernel, h_A.data(), h_B.data(), h_C.data(),
                                          M, N, K, grid_tc, block_tc, ITERATIONS);
        std::cout << std::setw(12) << tensorcore_ms;

        double cublas_ms = RunCuBLAS(cublas_handle, h_A.data(), h_B.data(), h_C.data(), M, N, K);
        std::cout << std::setw(12) << cublas_ms;

        dim3 block_cutlass(256, 1);
        dim3 grid_cutlass((N + 127) / 128, (M + 127) / 128);
        double cutlass_ms = RunKernel(GemmCutlassStyleKernel, h_A.data(), h_B.data(), h_C.data(),
                                       M, N, K, grid_cutlass, block_cutlass, ITERATIONS);
        std::cout << std::setw(12) << cutlass_ms << "\n";

        ofs << M << "," << N << "," << K << ","
            << naive_ms << "," << smem_ms << "," << opt_ms << "," << tensorcore_ms << ","
            << cublas_ms << "," << cutlass_ms << ","
            << ComputeGFLOPS(M, N, K, naive_ms) << ","
            << ComputeGFLOPS(M, N, K, smem_ms) << ","
            << ComputeGFLOPS(M, N, K, opt_ms) << ","
            << ComputeGFLOPS(M, N, K, tensorcore_ms) << ","
            << ComputeGFLOPS(M, N, K, cublas_ms) << ","
            << ComputeGFLOPS(M, N, K, cutlass_ms) << "\n";
    }

    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    std::cout << "\n========================================\n";
    std::cout << "Notes:\n";
    std::cout << "  - cuBLAS: NVIDIA's highly optimized BLAS library\n";
    std::cout << "  - Tensor Core: WMMA API for hardware acceleration\n";
    std::cout << "  - CUTLASS-style: Simplified CUTLASS-like tiling approach\n";
    std::cout << "  - For production, use official CUTLASS library with proper configuration\n";
    std::cout << "========================================\n";

    return 0;
}
