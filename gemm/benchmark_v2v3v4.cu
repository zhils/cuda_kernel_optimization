// GEMM V2, V3, V4 Performance Comparison Benchmark

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

#define CHECK_CUDA(call)                                                       \
  do {                                                                           \
    cudaError_t s__ = (call);                                                   \
    if (s__ != cudaSuccess) {                                                    \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "      \
                << cudaGetErrorString(s__) << std::endl;                         \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                            \
  } while (0)

// ============= V2 Kernel =============
namespace v2_config {
constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kTileK = 8;
constexpr int kTM = 8;
constexpr int kTN = 8;
constexpr int kBX = kBlockN / kTN;
constexpr int kBY = kBlockM / kTM;
}

__global__ void GemmV2Kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {
    using namespace v2_config;
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

// ============= V3 Kernel =============
namespace v3_config {
constexpr int kTileK = 8;
constexpr int kBX = 8;
constexpr int kBY = 8;
constexpr int kTM = 8;
constexpr int kTN = 8;
constexpr int kCtaM = kBY * kTM;
constexpr int kCtaN = kBX * kTN;
}

__global__ void GemmV3Kernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K) {
    using namespace v3_config;
    constexpr int NTHREADS = kBX * kBY;
    constexpr int A_LOADS = (kCtaM * (kTileK / 4)) / NTHREADS;
    constexpr int B_LOADS = (kTileK * (kCtaN / 4)) / NTHREADS;

    __shared__ float As[kCtaM][kTileK + 1];
    __shared__ float Bs[kCtaN][kTileK + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kBX + tx;
    const int row_start = blockIdx.y * kCtaM + ty * kTM;
    const int col_start = blockIdx.x * kCtaN + tx * kTN;

    float sum[kTM][kTN] = {};
    const int tiles = K / kTileK;

    float4 regA[A_LOADS];
    float4 regB[B_LOADS];

    #pragma unroll
    for (int li = 0; li < A_LOADS; ++li) {
        const int idx = tid + li * NTHREADS;
        const int row_idx = idx / (kTileK / 4);
        const int k4 = idx % (kTileK / 4);
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
            const int row_idx = idx / (kTileK / 4);
            const int k4 = idx % (kTileK / 4);
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
            const int next_k = (t + 1) * kTileK;
            #pragma unroll
            for (int li = 0; li < A_LOADS; ++li) {
                const int idx = tid + li * NTHREADS;
                const int row_idx = idx / (kTileK / 4);
                const int k4 = idx % (kTileK / 4);
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
        for (int kk = 0; kk < kTileK; ++kk) {
            float a[kTM], b[kTN];
            #pragma unroll
            for (int i = 0; i < kTM; ++i) a[i] = As[ty * kTM + i][kk];
            #pragma unroll
            for (int j = 0; j < kTN; ++j) b[j] = Bs[tx * kTN + j][kk];
            #pragma unroll
            for (int i = 0; i < kTM; ++i) {
                #pragma unroll
                for (int j = 0; j < kTN; ++j) {
                    sum[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < kTM; ++i) {
        float* row_ptr = C + (row_start + i) * N + col_start;
        #pragma unroll
        for (int j = 0; j < kTN; j += 4) {
            reinterpret_cast<float4*>(row_ptr + j)[0] =
                make_float4(sum[i][j + 0], sum[i][j + 1], sum[i][j + 2], sum[i][j + 3]);
        }
    }
}

// ============= V4 Kernel (FP16 TensorRT-style) =============
namespace v4_config {
constexpr int kTileK = 8;
constexpr int kBX = 8;
constexpr int kBY = 8;
constexpr int kTM = 8;
constexpr int kTN = 8;
constexpr int kCtaM = kBY * kTM;
constexpr int kCtaN = kBX * kTN;
}

__global__ void GemmV4Kernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K) {
    using namespace v4_config;
    constexpr int NTHREADS = kBX * kBY;
    constexpr int A_LOADS = (kCtaM * (kTileK / 4)) / NTHREADS;
    constexpr int B_LOADS = (kTileK * (kCtaN / 4)) / NTHREADS;

    __shared__ __half As[kCtaM][kTileK + 1];
    __shared__ __half Bs[kCtaN][kTileK + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kBX + tx;
    const int row_start = blockIdx.y * kCtaM + ty * kTM;
    const int col_start = blockIdx.x * kCtaN + tx * kTN;

    __half sum[kTM][kTN];
    #pragma unroll
    for (int i = 0; i < kTM; ++i)
        #pragma unroll
        for (int j = 0; j < kTN; ++j)
            sum[i][j] = __half(0.0f);

    const int tiles = K / kTileK;

    float4 regA[A_LOADS];
    float4 regB[B_LOADS];

    #pragma unroll
    for (int li = 0; li < A_LOADS; ++li) {
        const int idx = tid + li * NTHREADS;
        const int row_idx = idx / (kTileK / 4);
        const int k4 = idx % (kTileK / 4);
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
            const int row_idx = idx / (kTileK / 4);
            const int k4 = idx % (kTileK / 4);
            float4 v = regA[li];
            As[row_idx][k4 * 4 + 0] = __float2half(v.x);
            As[row_idx][k4 * 4 + 1] = __float2half(v.y);
            As[row_idx][k4 * 4 + 2] = __float2half(v.z);
            As[row_idx][k4 * 4 + 3] = __float2half(v.w);
        }
        #pragma unroll
        for (int li = 0; li < B_LOADS; ++li) {
            const int idx = tid + li * NTHREADS;
            const int k_idx = idx / (kCtaN / 4);
            const int n4 = idx % (kCtaN / 4);
            float4 v = regB[li];
            Bs[n4 * 4 + 0][k_idx] = __float2half(v.x);
            Bs[n4 * 4 + 1][k_idx] = __float2half(v.y);
            Bs[n4 * 4 + 2][k_idx] = __float2half(v.z);
            Bs[n4 * 4 + 3][k_idx] = __float2half(v.w);
        }
        __syncthreads();

        if (t + 1 < tiles) {
            const int next_k = (t + 1) * kTileK;
            #pragma unroll
            for (int li = 0; li < A_LOADS; ++li) {
                const int idx = tid + li * NTHREADS;
                const int row_idx = idx / (kTileK / 4);
                const int k4 = idx % (kTileK / 4);
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
        for (int kk = 0; kk < kTileK; ++kk) {
            __half a[kTM], b[kTN];
            #pragma unroll
            for (int i = 0; i < kTM; ++i) a[i] = As[ty * kTM + i][kk];
            #pragma unroll
            for (int j = 0; j < kTN; ++j) b[j] = Bs[tx * kTN + j][kk];
            #pragma unroll
            for (int i = 0; i < kTM; ++i) {
                #pragma unroll
                for (int j = 0; j < kTN; ++j) {
                    sum[i][j] = __hfma(a[i], b[j], sum[i][j]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < kTM; ++i) {
        float* row_ptr = C + (row_start + i) * N + col_start;
        #pragma unroll
        for (int j = 0; j < kTN; j += 4) {
            float4 out;
            out.x = __half2float(sum[i][j + 0]);
            out.y = __half2float(sum[i][j + 1]);
            out.z = __half2float(sum[i][j + 2]);
            out.w = __half2float(sum[i][j + 3]);
            reinterpret_cast<float4*>(row_ptr + j)[0] = out;
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
    std::cout << "  GEMM V2, V3, V4 Performance Comparison\n";
    std::cout << "=======================================================\n\n";

    std::vector<std::tuple<int, int, int>> test_cases = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
    };

    std::cout << std::left
              << std::setw(8) << "M"
              << std::setw(8) << "N"
              << std::setw(8) << "K"
              << std::setw(14) << "V2 (ms)"
              << std::setw(14) << "V3 (ms)"
              << std::setw(14) << "V4 (ms)"
              << std::setw(14) << "V3 vs V2"
              << std::setw(14) << "V4 vs V2"
              << "\n";
    std::cout << std::string(80, '-') << "\n";

    std::ofstream ofs("data/results/gemm_v2v3v4_comparison.csv");
    ofs << "M,N,K,v2_ms,v3_ms,v4_ms,v2_gflops,v3_gflops,v4_gflops,v3_speedup,v4_speedup\n";

    dim3 block_v2(v2_config::kBX, v2_config::kBY);
    dim3 block_v3(v3_config::kBX, v3_config::kBY);
    dim3 block_v4(v4_config::kBX, v4_config::kBY);

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

        double ms_v2 = 0.0, ms_v3 = 0.0, ms_v4 = 0.0;

        {
            dim3 grid_v2((N + v2_config::kBlockN - 1) / v2_config::kBlockN,
                         (M + v2_config::kBlockM - 1) / v2_config::kBlockM);
            ms_v2 = MeasureKernel([&]() {
                GemmV2Kernel<<<grid_v2, block_v2>>>(dA, dB, dC, M, N, K);
            }, kIterations);
        }

        {
            dim3 grid_v3((N + v3_config::kCtaN - 1) / v3_config::kCtaN,
                         (M + v3_config::kCtaM - 1) / v3_config::kCtaM);
            ms_v3 = MeasureKernel([&]() {
                GemmV3Kernel<<<grid_v3, block_v3>>>(dA, dB, dC, M, N, K);
            }, kIterations);
        }

        {
            dim3 grid_v4((N + v4_config::kCtaN - 1) / v4_config::kCtaN,
                         (M + v4_config::kCtaM - 1) / v4_config::kCtaM);
            ms_v4 = MeasureKernel([&]() {
                GemmV4Kernel<<<grid_v4, block_v4>>>(dA, dB, dC, M, N, K);
            }, kIterations);
        }

        CHECK_CUDA(cudaFree(dA));
        CHECK_CUDA(cudaFree(dB));
        CHECK_CUDA(cudaFree(dC));

        double speedup_v3 = (ms_v2 > 0) ? ms_v2 / ms_v3 : 1.0;
        double speedup_v4 = (ms_v2 > 0) ? ms_v2 / ms_v4 : 1.0;

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(14) << ms_v2
                  << std::setw(14) << ms_v3
                  << std::setw(14) << ms_v4
                  << std::setw(14) << speedup_v3 << "x"
                  << std::setw(14) << speedup_v4 << "x";

        std::cout << "\n";

        ofs << M << "," << N << "," << K << ","
            << ms_v2 << "," << ms_v3 << "," << ms_v4 << ","
            << ComputeGFLOPS(M, N, K, ms_v2) << ","
            << ComputeGFLOPS(M, N, K, ms_v3) << ","
            << ComputeGFLOPS(M, N, K, ms_v4) << ","
            << speedup_v3 << "," << speedup_v4 << "\n";
    }

    std::cout << "\n=======================================================\n";
    std::cout << "  V2: Register tiling (TM=TN=8)\n";
    std::cout << "  V3: Software pipelining + prefetch\n";
    std::cout << "  V4: FP16 TensorRT-style with __hfma\n";
    std::cout << "=======================================================\n";

    return 0;
}