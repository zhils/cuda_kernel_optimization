// GEMM Performance Comparison: V0 Naive | V1 SMEM Tiling | V2 (same as gemm_v2.cu) | V3 (same as gemm_v3.cu) | cuBLAS FP32

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
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

// ============ V0: Naive GEMM ============
__global__ void GemmV0NaiveKernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ============ V1: Shared Memory Tiling ============
namespace {
constexpr int kTile = 16;
constexpr int kTileK = 32;
}

__global__ void GemmV1Kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {
    __shared__ float As[kTile][kTileK + 1];
    __shared__ float Bs[kTileK][kTile + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kTile + tx;
    const int row = blockIdx.y * kTile + ty;
    const int col = blockIdx.x * kTile + tx;

    const bool can_vec_a = (K % 4 == 0);
    const bool can_vec_b = (N % 4 == 0);

    float sum = 0.0f;
    const int k_tiles = (K + kTileK - 1) / kTileK;

    for (int t = 0; t < k_tiles; ++t) {
        const int k0 = t * kTileK;

        if (tid < kTile * (kTileK / 4)) {
            const int a_row = tid / (kTileK / 4);
            const int a_k4  = tid % (kTileK / 4);
            const int g_row = blockIdx.y * kTile + a_row;
            const int g_col = k0 + a_k4 * 4;

            if (can_vec_a && g_row < M && g_col + 3 < K) {
                const float4 v = __ldg(reinterpret_cast<const float4*>(
                                     A + g_row * K + g_col));
                As[a_row][a_k4 * 4 + 0] = v.x;
                As[a_row][a_k4 * 4 + 1] = v.y;
                As[a_row][a_k4 * 4 + 2] = v.z;
                As[a_row][a_k4 * 4 + 3] = v.w;
            } else {
                for (int d = 0; d < 4; ++d) {
                    const int gc = g_col + d;
                    As[a_row][a_k4 * 4 + d] =
                        (g_row < M && gc < K) ? __ldg(A + g_row * K + gc) : 0.0f;
                }
            }
        } else {
            const int b_tid = tid - kTile * (kTileK / 4);
            const int b_row = b_tid / (kTile / 4);
            const int b_n4  = b_tid % (kTile / 4);
            const int g_row = k0 + b_row;
            const int g_col = blockIdx.x * kTile + b_n4 * 4;

            if (can_vec_b && g_row < K && g_col + 3 < N) {
                const float4 v = __ldg(reinterpret_cast<const float4*>(
                                     B + g_row * N + g_col));
                Bs[b_row][b_n4 * 4 + 0] = v.x;
                Bs[b_row][b_n4 * 4 + 1] = v.y;
                Bs[b_row][b_n4 * 4 + 2] = v.z;
                Bs[b_row][b_n4 * 4 + 3] = v.w;
            } else {
                for (int d = 0; d < 4; ++d) {
                    const int gc = g_col + d;
                    Bs[b_row][b_n4 * 4 + d] =
                        (g_row < K && gc < N) ? __ldg(B + g_row * N + gc) : 0.0f;
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            sum += As[ty][kk] * Bs[kk][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// V2 / V3 kernels are shared with gemm_v2.cu and gemm_v3.cu (single source by include).
#include "gemm_benchmark_v2_v3.cuh"

// ============ Benchmark Helpers ============
template <typename LaunchFn>
double MeasureKernel(LaunchFn launch, int iterations) {
    launch();
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> times;
    times.reserve(iterations);
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < iterations; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        launch();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    std::sort(times.begin(), times.end());
    if (times.size() > 2) {
        double sum = 0.0;
        for (size_t i = 1; i + 1 < times.size(); ++i) sum += times[i];
        return sum / static_cast<double>(times.size() - 2);
    }
    double sum = 0.0;
    for (float t : times) sum += t;
    return sum / static_cast<double>(times.size());
}

static double ComputeGFLOPS(int M, int N, int K, double ms) {
    if (ms <= 0) return 0;
    return 2.0 * M * N * K / (ms * 1e6);
}

// ============ Main ============
int main() {
    constexpr int kIterations = 10;

    std::cout << "========================================\n";
    std::cout << "  GEMM Performance Comparison\n";
    std::cout << "  V0 Naive | V1 SMEM Tiling | V2 RegTile (gemm_v2) | V3 DblBuf (gemm_v3) | cuBLAS FP32\n";
    std::cout << "========================================\n\n";

    std::vector<std::tuple<int, int, int>> test_cases = {
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
    };

    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/gemm_all_comparison.csv");
    ofs << "M,N,K,v0_ms,v1_ms,v2_ms,v3_ms,cublas_ms,v0_gflops,v1_gflops,v2_gflops,v3_gflops,cublas_gflops\n";

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    std::cout << std::left
              << std::setw(6) << "M"
              << std::setw(6) << "N"
              << std::setw(6) << "K"
              << std::setw(12) << "V0 Naive"
              << std::setw(12) << "V1 SMEM"
              << std::setw(12) << "V2 Reg"
              << std::setw(12) << "V3 DblBuf"
              << std::setw(12) << "cuBLAS"
              << "\n";
    std::cout << std::string(66, '-') << "\n";

    for (const auto& [M, N, K] : test_cases) {
        std::vector<float> h_A(static_cast<size_t>(M) * K),
                           h_B(static_cast<size_t>(K) * N),
                           h_C(static_cast<size_t>(M) * N);
        common::InitMatrix(h_A, M, K);
        common::InitMatrix(h_B, K, N);

        float *dA, *dB, *dC;
        CHECK_CUDA(cudaMalloc(&dA, h_A.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dB, h_B.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dC, h_C.size() * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(dA, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));

        std::cout << std::left
                  << std::setw(6) << M
                  << std::setw(6) << N
                  << std::setw(6) << K;

        double v0_ms = 0.0, v1_ms = 0.0, v2_ms = 0.0, v3_ms = 0.0, cublas_ms = 0.0;

        {
            dim3 block(16, 16), grid((N + 15) / 16, (M + 15) / 16);
            v0_ms = MeasureKernel([&]() {
                GemmV0NaiveKernel<<<grid, block>>>(dA, dB, dC, M, N, K);
            }, kIterations);
        }

        {
            dim3 block(kTile, kTile), grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
            v1_ms = MeasureKernel([&]() {
                GemmV1Kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
            }, kIterations);
        }

        {
            dim3 block(gemm_bench_v2::kBlockThreadsX, gemm_bench_v2::kBlockThreadsY),
                grid((N + gemm_bench_v2::kBlockN - 1) / gemm_bench_v2::kBlockN,
                     (M + gemm_bench_v2::kBlockM - 1) / gemm_bench_v2::kBlockM);
            v2_ms = MeasureKernel([&]() {
                gemm_bench_v2::GemmV2Kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
            }, kIterations);
        }

        {
            const bool aligned = (M % gemm_bench_v3::kBlockM == 0) &&
                                 (N % gemm_bench_v3::kBlockN == 0) &&
                                 (K % gemm_bench_v3::kTileK == 0);
            if (aligned) {
                dim3 block(gemm_bench_v3::kBlockThreadsX, gemm_bench_v3::kBlockThreadsY),
                    grid((N + gemm_bench_v3::kBlockN - 1) / gemm_bench_v3::kBlockN,
                         (M + gemm_bench_v3::kBlockM - 1) / gemm_bench_v3::kBlockM);
                v3_ms = MeasureKernel([&]() {
                    gemm_bench_v3::GemmV3Kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
                }, kIterations);
            }
        }

        {
            float alpha = 1.0f, beta = 0.0f;
            cublas_ms = MeasureKernel([&]() {
                CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
            }, kIterations);
        }

        CHECK_CUDA(cudaFree(dA));
        CHECK_CUDA(cudaFree(dB));
        CHECK_CUDA(cudaFree(dC));

        auto print_cell = [](double ms) {
            if (ms <= 0.0)
                std::cout << std::setw(12) << "---";
            else
                std::cout << std::fixed << std::setprecision(3) << std::setw(12) << ms;
        };
        print_cell(v0_ms);
        print_cell(v1_ms);
        print_cell(v2_ms);
        print_cell(v3_ms);
        print_cell(cublas_ms);
        std::cout << "\n";

        ofs << M << "," << N << "," << K << ","
            << v0_ms << "," << v1_ms << "," << v2_ms << "," << v3_ms << "," << cublas_ms << ","
            << ComputeGFLOPS(M, N, K, v0_ms) << ","
            << ComputeGFLOPS(M, N, K, v1_ms) << ","
            << ComputeGFLOPS(M, N, K, v2_ms) << ","
            << ComputeGFLOPS(M, N, K, v3_ms) << ","
            << ComputeGFLOPS(M, N, K, cublas_ms) << "\n";
    }

    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    std::cout << "\n========================================\n";
    std::cout << "  V0: One thread per output element, global memory only\n";
    std::cout << "  V1: Shared memory tiling + float4 vectorized + __ldg\n";
    std::cout << "  V2: Same kernel as gemm_v2.cu - 16x16 threads, TM=TN=8, CTA 128x128, TileK=16\n";
    std::cout << "  V3: Same kernel as gemm_v3.cu - double-buffered SMEM, same geometry as V2\n";
    std::cout << "  cuBLAS: cublasSgemm FP32 (implementation may use Tensor Math on supported GPUs)\n";
    std::cout << "========================================\n";

    return 0;
}
