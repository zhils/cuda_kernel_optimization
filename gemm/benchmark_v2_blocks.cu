// GEMM V2 Register Tiling Benchmark: Testing different TM×TN block sizes
// Comparing: 4×8, 8×8, 8×16, 16×16

#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

// ============= 4x8 Configuration =============
namespace config_4x8 {
constexpr int kBlockM = 32;
constexpr int kBlockN = 64;
constexpr int kTileK = 8;
constexpr int kTM = 4;
constexpr int kTN = 8;
constexpr int kBX = kBlockN / kTN;
constexpr int kBY = kBlockM / kTM;
}

__global__ void GemmV2Kernel_4x8(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    using namespace config_4x8;
    __shared__ float As[kBlockM][kTileK + 1];
    __shared__ float Bs[kBlockN][kTileK + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kBX + tx;
    const int row_start = blockIdx.y * kBlockM + ty * kTM;
    const int col_start = blockIdx.x * kBlockN + tx * kTN;

    const bool can_vec_a = (K % 4 == 0);
    const bool can_vec_b = (N % 4 == 0);

    float sum[kTM][kTN] = {};
    const int k_tiles = (K + kTileK - 1) / kTileK;

    for (int t = 0; t < k_tiles; ++t) {
        const int k0 = t * kTileK;

        {
            const int a_row = tid / (kTileK / 4);
            const int a_k4  = tid % (kTileK / 4);
            const int g_row = blockIdx.y * kBlockM + a_row;
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
        }

        {
            const int b_k  = tid / (kBlockN / 4);
            const int b_n4 = tid % (kBlockN / 4);
            const int g_row = k0 + b_k;
            const int g_col = blockIdx.x * kBlockN + b_n4 * 4;

            if (can_vec_b && g_row < K && g_col + 3 < N) {
                const float4 v = __ldg(reinterpret_cast<const float4*>(
                                     B + g_row * N + g_col));
                Bs[b_n4 * 4 + 0][b_k] = v.x;
                Bs[b_n4 * 4 + 1][b_k] = v.y;
                Bs[b_n4 * 4 + 2][b_k] = v.z;
                Bs[b_n4 * 4 + 3][b_k] = v.w;
            } else {
                for (int d = 0; d < 4; ++d) {
                    const int gc = g_col + d;
                    Bs[b_n4 * 4 + d][b_k] =
                        (g_row < K && gc < N) ? __ldg(B + g_row * N + gc) : 0.0f;
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            float b[kTN];
            #pragma unroll
            for (int j = 0; j < kTN; ++j) b[j] = Bs[tx * kTN + j][kk];
            #pragma unroll
            for (int i = 0; i < kTM; ++i) {
                const float a = As[ty * kTM + i][kk];
                #pragma unroll
                for (int j = 0; j < kTN; ++j) {
                    sum[i][j] += a * b[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < kTM; ++i) {
        const int r = row_start + i;
        #pragma unroll
        for (int j = 0; j < kTN; ++j) {
            const int c = col_start + j;
            if (r < M && c < N) C[r * N + c] = sum[i][j];
        }
    }
}

// ============= 8x8 Configuration =============
namespace config_8x8 {
constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kTileK = 8;
constexpr int kTM = 8;
constexpr int kTN = 8;
constexpr int kBX = kBlockN / kTN;
constexpr int kBY = kBlockM / kTM;
}

__global__ void GemmV2Kernel_8x8(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    using namespace config_8x8;
    __shared__ float As[kBlockM][kTileK + 1];
    __shared__ float Bs[kBlockN][kTileK + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kBX + tx;
    const int row_start = blockIdx.y * kBlockM + ty * kTM;
    const int col_start = blockIdx.x * kBlockN + tx * kTN;

    const bool can_vec_a = (K % 4 == 0);
    const bool can_vec_b = (N % 4 == 0);

    float sum[kTM][kTN] = {};
    const int k_tiles = (K + kTileK - 1) / kTileK;

    for (int t = 0; t < k_tiles; ++t) {
        const int k0 = t * kTileK;

        {
            const int a_row = tid / (kTileK / 4);
            const int a_k4  = tid % (kTileK / 4);
            const int g_row = blockIdx.y * kBlockM + a_row;
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
        }

        {
            const int b_k  = tid / (kBlockN / 4);
            const int b_n4 = tid % (kBlockN / 4);
            const int g_row = k0 + b_k;
            const int g_col = blockIdx.x * kBlockN + b_n4 * 4;

            if (can_vec_b && g_row < K && g_col + 3 < N) {
                const float4 v = __ldg(reinterpret_cast<const float4*>(
                                     B + g_row * N + g_col));
                Bs[b_n4 * 4 + 0][b_k] = v.x;
                Bs[b_n4 * 4 + 1][b_k] = v.y;
                Bs[b_n4 * 4 + 2][b_k] = v.z;
                Bs[b_n4 * 4 + 3][b_k] = v.w;
            } else {
                for (int d = 0; d < 4; ++d) {
                    const int gc = g_col + d;
                    Bs[b_n4 * 4 + d][b_k] =
                        (g_row < K && gc < N) ? __ldg(B + g_row * N + gc) : 0.0f;
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            float b[kTN];
            #pragma unroll
            for (int j = 0; j < kTN; ++j) b[j] = Bs[tx * kTN + j][kk];
            #pragma unroll
            for (int i = 0; i < kTM; ++i) {
                const float a = As[ty * kTM + i][kk];
                #pragma unroll
                for (int j = 0; j < kTN; ++j) {
                    sum[i][j] += a * b[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < kTM; ++i) {
        const int r = row_start + i;
        #pragma unroll
        for (int j = 0; j < kTN; ++j) {
            const int c = col_start + j;
            if (r < M && c < N) C[r * N + c] = sum[i][j];
        }
    }
}

// ============= 8x16 Configuration =============
namespace config_8x16 {
constexpr int kBlockM = 64;
constexpr int kBlockN = 128;
constexpr int kTileK = 8;
constexpr int kTM = 8;
constexpr int kTN = 16;
constexpr int kBX = kBlockN / kTN;
constexpr int kBY = kBlockM / kTM;
}

__global__ void GemmV2Kernel_8x16(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    using namespace config_8x16;
    __shared__ float As[kBlockM][kTileK + 1];
    __shared__ float Bs[kBlockN][kTileK + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kBX + tx;
    const int row_start = blockIdx.y * kBlockM + ty * kTM;
    const int col_start = blockIdx.x * kBlockN + tx * kTN;

    const bool can_vec_a = (K % 4 == 0);
    const bool can_vec_b = (N % 4 == 0);

    float sum[kTM][kTN] = {};
    const int k_tiles = (K + kTileK - 1) / kTileK;

    for (int t = 0; t < k_tiles; ++t) {
        const int k0 = t * kTileK;

        {
            const int a_row = tid / (kTileK / 4);
            const int a_k4  = tid % (kTileK / 4);
            const int g_row = blockIdx.y * kBlockM + a_row;
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
        }

        {
            const int b_k  = tid / (kBlockN / 4);
            const int b_n4 = tid % (kBlockN / 4);
            const int g_row = k0 + b_k;
            const int g_col = blockIdx.x * kBlockN + b_n4 * 4;

            if (can_vec_b && g_row < K && g_col + 3 < N) {
                const float4 v = __ldg(reinterpret_cast<const float4*>(
                                     B + g_row * N + g_col));
                Bs[b_n4 * 4 + 0][b_k] = v.x;
                Bs[b_n4 * 4 + 1][b_k] = v.y;
                Bs[b_n4 * 4 + 2][b_k] = v.z;
                Bs[b_n4 * 4 + 3][b_k] = v.w;
            } else {
                for (int d = 0; d < 4; ++d) {
                    const int gc = g_col + d;
                    Bs[b_n4 * 4 + d][b_k] =
                        (g_row < K && gc < N) ? __ldg(B + g_row * N + gc) : 0.0f;
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            float b[kTN];
            #pragma unroll
            for (int j = 0; j < kTN; ++j) b[j] = Bs[tx * kTN + j][kk];
            #pragma unroll
            for (int i = 0; i < kTM; ++i) {
                const float a = As[ty * kTM + i][kk];
                #pragma unroll
                for (int j = 0; j < kTN; ++j) {
                    sum[i][j] += a * b[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < kTM; ++i) {
        const int r = row_start + i;
        #pragma unroll
        for (int j = 0; j < kTN; ++j) {
            const int c = col_start + j;
            if (r < M && c < N) C[r * N + c] = sum[i][j];
        }
    }
}

// ============= 16x16 Configuration =============
namespace config_16x16 {
constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kTileK = 8;
constexpr int kTM = 16;
constexpr int kTN = 16;
constexpr int kBX = kBlockN / kTN;
constexpr int kBY = kBlockM / kTM;
}

__global__ void GemmV2Kernel_16x16(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K) {
    using namespace config_16x16;
    __shared__ float As[kBlockM][kTileK + 1];
    __shared__ float Bs[kBlockN][kTileK + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kBX + tx;
    const int row_start = blockIdx.y * kBlockM + ty * kTM;
    const int col_start = blockIdx.x * kBlockN + tx * kTN;

    const bool can_vec_a = (K % 4 == 0);
    const bool can_vec_b = (N % 4 == 0);

    float sum[kTM][kTN] = {};
    const int k_tiles = (K + kTileK - 1) / kTileK;

    for (int t = 0; t < k_tiles; ++t) {
        const int k0 = t * kTileK;

        {
            const int a_row = tid / (kTileK / 4);
            const int a_k4  = tid % (kTileK / 4);
            const int g_row = blockIdx.y * kBlockM + a_row;
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
        }

        {
            const int b_k  = tid / (kBlockN / 4);
            const int b_n4 = tid % (kBlockN / 4);
            const int g_row = k0 + b_k;
            const int g_col = blockIdx.x * kBlockN + b_n4 * 4;

            if (can_vec_b && g_row < K && g_col + 3 < N) {
                const float4 v = __ldg(reinterpret_cast<const float4*>(
                                     B + g_row * N + g_col));
                Bs[b_n4 * 4 + 0][b_k] = v.x;
                Bs[b_n4 * 4 + 1][b_k] = v.y;
                Bs[b_n4 * 4 + 2][b_k] = v.z;
                Bs[b_n4 * 4 + 3][b_k] = v.w;
            } else {
                for (int d = 0; d < 4; ++d) {
                    const int gc = g_col + d;
                    Bs[b_n4 * 4 + d][b_k] =
                        (g_row < K && gc < N) ? __ldg(B + g_row * N + gc) : 0.0f;
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            float b[kTN];
            #pragma unroll
            for (int j = 0; j < kTN; ++j) b[j] = Bs[tx * kTN + j][kk];
            #pragma unroll
            for (int i = 0; i < kTM; ++i) {
                const float a = As[ty * kTM + i][kk];
                #pragma unroll
                for (int j = 0; j < kTN; ++j) {
                    sum[i][j] += a * b[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < kTM; ++i) {
        const int r = row_start + i;
        #pragma unroll
        for (int j = 0; j < kTN; ++j) {
            const int c = col_start + j;
            if (r < M && c < N) C[r * N + c] = sum[i][j];
        }
    }
}

// ============= Benchmark Helpers =============
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

int main() {
    constexpr int kIterations = 10;

    std::cout << "=======================================================\n";
    std::cout << "  GEMM V2 Register Tiling Block Size Comparison\n";
    std::cout << "  Testing: 4x8, 8x8, 8x16, 16x16\n";
    std::cout << "=======================================================\n\n";

    std::vector<std::tuple<int, int, int>> test_cases = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
    };

    std::cout << std::left
              << std::setw(8) << "M"
              << std::setw(8) << "N"
              << std::setw(8) << "K"
              << std::setw(12) << "4x8 ms"
              << std::setw(12) << "8x8 ms"
              << std::setw(12) << "8x16 ms"
              << std::setw(12) << "16x16 ms"
              << "\n";
    std::cout << std::string(60, '-') << "\n";

    std::ofstream ofs("data/results/gemm_v2_block_comparison.csv");
    ofs << "M,N,K,block_4x8_ms,block_8x8_ms,block_8x16_ms,block_16x16_ms,"
        << "block_4x8_gflops,block_8x8_gflops,block_8x16_gflops,block_16x16_gflops\n";

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
                  << std::setw(8) << M
                  << std::setw(8) << N
                  << std::setw(8) << K;

        double ms_4x8 = 0, ms_8x8 = 0, ms_8x16 = 0, ms_16x16 = 0;

        {
            dim3 block_4x8(config_4x8::kBX, config_4x8::kBY);
            dim3 grid_4x8((N + config_4x8::kBlockN - 1) / config_4x8::kBlockN,
                          (M + config_4x8::kBlockM - 1) / config_4x8::kBlockM);
            ms_4x8 = MeasureKernel([&]() {
                GemmV2Kernel_4x8<<<grid_4x8, block_4x8>>>(dA, dB, dC, M, N, K);
            }, kIterations);
        }

        {
            dim3 block_8x8(config_8x8::kBX, config_8x8::kBY);
            dim3 grid_8x8((N + config_8x8::kBlockN - 1) / config_8x8::kBlockN,
                          (M + config_8x8::kBlockM - 1) / config_8x8::kBlockM);
            ms_8x8 = MeasureKernel([&]() {
                GemmV2Kernel_8x8<<<grid_8x8, block_8x8>>>(dA, dB, dC, M, N, K);
            }, kIterations);
        }

        {
            dim3 block_8x16(config_8x16::kBX, config_8x16::kBY);
            dim3 grid_8x16((N + config_8x16::kBlockN - 1) / config_8x16::kBlockN,
                           (M + config_8x16::kBlockM - 1) / config_8x16::kBlockM);
            ms_8x16 = MeasureKernel([&]() {
                GemmV2Kernel_8x16<<<grid_8x16, block_8x16>>>(dA, dB, dC, M, N, K);
            }, kIterations);
        }

        {
            dim3 block_16x16(config_16x16::kBX, config_16x16::kBY);
            dim3 grid_16x16((N + config_16x16::kBlockN - 1) / config_16x16::kBlockN,
                            (M + config_16x16::kBlockM - 1) / config_16x16::kBlockM);
            ms_16x16 = MeasureKernel([&]() {
                GemmV2Kernel_16x16<<<grid_16x16, block_16x16>>>(dA, dB, dC, M, N, K);
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
        print_cell(ms_4x8);
        print_cell(ms_8x8);
        print_cell(ms_8x16);
        print_cell(ms_16x16);
        std::cout << "\n";

        ofs << M << "," << N << "," << K << ","
            << ms_4x8 << "," << ms_8x8 << "," << ms_8x16 << "," << ms_16x16 << ","
            << ComputeGFLOPS(M, N, K, ms_4x8) << ","
            << ComputeGFLOPS(M, N, K, ms_8x8) << ","
            << ComputeGFLOPS(M, N, K, ms_8x16) << ","
            << ComputeGFLOPS(M, N, K, ms_16x16) << "\n";
    }

    std::cout << "\n=======================================================\n";
    std::cout << "  Block Size Configuration Summary:\n";
    std::cout << "  4x8 : TM=4, TN=8, threads=32, blockM=32, blockN=64\n";
    std::cout << "  8x8 : TM=8, TN=8, threads=64, blockM=64, blockN=64\n";
    std::cout << "  8x16: TM=8, TN=16, threads=128, blockM=64, blockN=128\n";
    std::cout << "  16x16: TM=16, TN=16, threads=16, blockM=64, blockN=64\n";
    std::cout << "=======================================================\n";

    return 0;
}