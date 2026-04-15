// GEMM Performance Comparison: V0 Naive | V1 SMEM Tiling | V2 Register Tiling | V3 Prefetch | cuBLAS

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

// ============ V2: Thread-level Register Tiling ============
namespace {
constexpr int kBlockM = 32;
constexpr int kBlockN = 32;
constexpr int kTileK_V2 = 8;
constexpr int kTM = 4;
constexpr int kTN = 4;
constexpr int kBX = kBlockN / kTN;
constexpr int kBY = kBlockM / kTM;
}

__global__ void GemmV2Kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {
    __shared__ float As[kBlockM][kTileK_V2 + 1];
    __shared__ float Bs[kBlockN][kTileK_V2 + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kBX + tx;
    const int row_start = blockIdx.y * kBlockM + ty * kTM;
    const int col_start = blockIdx.x * kBlockN + tx * kTN;

    const bool can_vec_a = (K % 4 == 0);
    const bool can_vec_b = (N % 4 == 0);

    float sum[kTM][kTN] = {};

    const int k_tiles = (K + kTileK_V2 - 1) / kTileK_V2;

    for (int t = 0; t < k_tiles; ++t) {
        const int k0 = t * kTileK_V2;

        {
            const int a_row = tid / (kTileK_V2 / 4);
            const int a_k4  = tid % (kTileK_V2 / 4);
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
        for (int kk = 0; kk < kTileK_V2; ++kk) {
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

// ============ V3: Register Prefetch (Software Pipelining) ============
namespace {
constexpr int kTileK_V3 = 32;
constexpr int kBX_V3 = 16;
constexpr int kBY_V3 = 16;
constexpr int kTM_V3 = 8;
constexpr int kTN_V3 = 8;
constexpr int kCtaM = kBY_V3 * kTM_V3;
constexpr int kCtaN = kBX_V3 * kTN_V3;
}

__global__ void __launch_bounds__(256, 2)
GemmV3Kernel(const float* __restrict__ A,
             const float* __restrict__ B,
             float* __restrict__ C,
             int M, int N, int K) {
    constexpr int NTHREADS = kBX_V3 * kBY_V3;
    constexpr int A_LOADS = (kCtaM * (kTileK_V3 / 4)) / NTHREADS;
    constexpr int B_LOADS = (kTileK_V3 * (kCtaN / 4)) / NTHREADS;

    __shared__ float As[kCtaM][kTileK_V3 + 1];
    __shared__ float Bs[kCtaN][kTileK_V3 + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kBX_V3 + tx;
    const int row_start = blockIdx.y * kCtaM + ty * kTM_V3;
    const int col_start = blockIdx.x * kCtaN + tx * kTN_V3;

    float sum[kTM_V3][kTN_V3] = {};
    const int tiles = K / kTileK_V3;

    float4 regA[A_LOADS];
    float4 regB[B_LOADS];

    #pragma unroll
    for (int li = 0; li < A_LOADS; ++li) {
        const int idx = tid + li * NTHREADS;
        const int row_idx = idx / (kTileK_V3 / 4);
        const int k4 = idx % (kTileK_V3 / 4);
        regA[li] = reinterpret_cast<const float4*>(
                       A + (blockIdx.y * kCtaM + row_idx) * K)[k4];
    }
    #pragma unroll
    for (int li = 0; li < B_LOADS; ++li) {
        const int idx = tid + li * NTHREADS;
        const int k_idx = idx / (kCtaN / 4);
        const int n4 = idx % (kCtaN / 4);
        regB[li] = reinterpret_cast<const float4*>(
                       B + k_idx * N + blockIdx.x * kCtaN + n4 * 4)[0];
    }

    for (int t = 0; t < tiles; ++t) {
        #pragma unroll
        for (int li = 0; li < A_LOADS; ++li) {
            const int idx = tid + li * NTHREADS;
            const int row_idx = idx / (kTileK_V3 / 4);
            const int k4 = idx % (kTileK_V3 / 4);
            As[row_idx][k4 * 4 + 0] = regA[li].x;
            As[row_idx][k4 * 4 + 1] = regA[li].y;
            As[row_idx][k4 * 4 + 2] = regA[li].z;
            As[row_idx][k4 * 4 + 3] = regA[li].w;
        }
        #pragma unroll
        for (int li = 0; li < B_LOADS; ++li) {
            const int idx = tid + li * NTHREADS;
            const int k_idx = idx / (kCtaN / 4);
            const int n4 = idx % (kCtaN / 4);
            Bs[n4 * 4 + 0][k_idx] = regB[li].x;
            Bs[n4 * 4 + 1][k_idx] = regB[li].y;
            Bs[n4 * 4 + 2][k_idx] = regB[li].z;
            Bs[n4 * 4 + 3][k_idx] = regB[li].w;
        }
        __syncthreads();

        if (t + 1 < tiles) {
            const int next_k = (t + 1) * kTileK_V3;
            #pragma unroll
            for (int li = 0; li < A_LOADS; ++li) {
                const int idx = tid + li * NTHREADS;
                const int row_idx = idx / (kTileK_V3 / 4);
                const int k4 = idx % (kTileK_V3 / 4);
                regA[li] = reinterpret_cast<const float4*>(
                               A + (blockIdx.y * kCtaM + row_idx) * K + next_k)[k4];
            }
            #pragma unroll
            for (int li = 0; li < B_LOADS; ++li) {
                const int idx = tid + li * NTHREADS;
                const int k_idx = idx / (kCtaN / 4);
                const int n4 = idx % (kCtaN / 4);
                regB[li] = reinterpret_cast<const float4*>(
                               B + (next_k + k_idx) * N + blockIdx.x * kCtaN + n4 * 4)[0];
            }
        }

        #pragma unroll
        for (int kk = 0; kk < kTileK_V3; ++kk) {
            float a[kTM_V3], b[kTN_V3];
            #pragma unroll
            for (int i = 0; i < kTM_V3; ++i) a[i] = As[ty * kTM_V3 + i][kk];
            #pragma unroll
            for (int j = 0; j < kTN_V3; ++j) b[j] = Bs[tx * kTN_V3 + j][kk];
            #pragma unroll
            for (int i = 0; i < kTM_V3; ++i) {
                #pragma unroll
                for (int j = 0; j < kTN_V3; ++j) {
                    sum[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < kTM_V3; ++i) {
        float* row_ptr = C + (row_start + i) * N + col_start;
        #pragma unroll
        for (int j = 0; j < kTN_V3; j += 4) {
            reinterpret_cast<float4*>(row_ptr + j)[0] =
                make_float4(sum[i][j + 0], sum[i][j + 1], sum[i][j + 2], sum[i][j + 3]);
        }
    }
}

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
    std::cout << "  V0 Naive | V1 SMEM Tiling | V2 Register Tiling | V3 Prefetch | cuBLAS\n";
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
              << std::setw(12) << "V3 Prefetch"
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
            dim3 block(kBX, kBY), grid((N + kBlockN - 1) / kBlockN, (M + kBlockM - 1) / kBlockM);
            v2_ms = MeasureKernel([&]() {
                GemmV2Kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
            }, kIterations);
        }

        {
            const bool aligned = (M % kCtaM == 0) && (N % kCtaN == 0) && (K % kTileK_V3 == 0);
            if (aligned) {
                dim3 block(kBX_V3, kBY_V3), grid((N + kCtaN - 1) / kCtaN, (M + kCtaM - 1) / kCtaM);
                v3_ms = MeasureKernel([&]() {
                    GemmV3Kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
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
    std::cout << "  V2: Thread-level register tiling (TM=TN=4)\n";
    std::cout << "  V3: Register prefetch (software pipelining)\n";
    std::cout << "  cuBLAS: NVIDIA cublasSgemm (may use TF32 Tensor Core)\n";
    std::cout << "========================================\n";

    return 0;
}
