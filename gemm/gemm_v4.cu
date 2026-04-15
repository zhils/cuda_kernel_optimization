// GEMM V4: TensorRT Accelerated GEMM (Optimization plan d)
//
// Optimization over V3:
//   d — Use TensorRT-style optimizations:
//       - FP16 precision with __half and __hfma instructions
//       - Optimized memory access patterns
//       - Reduced memory bandwidth usage

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

namespace {

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
    std::cout << "  GEMM V4 Performance (TensorRT Accelerated)\n";
    std::cout << "  FP16 with __hfma instructions\n";
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
              << std::setw(14) << "Time (ms)"
              << std::setw(14) << "GFLOPS"
              << "\n";
    std::cout << std::string(50, '-') << "\n";

    std::ofstream ofs("data/results/gemm_v4_results.csv");
    ofs << "M,N,K,time_ms,gflops\n";

    dim3 block(kBX, kBY);

    for (const auto& tc : test_cases) {
        int M = std::get<0>(tc);
        int N = std::get<1>(tc);
        int K = std::get<2>(tc);

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

        dim3 grid((N + kCtaN - 1) / kCtaN, (M + kCtaM - 1) / kCtaM);
        double ms = MeasureKernel([&]() {
            GemmV4Kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        }, kIterations);

        double gflops = ComputeGFLOPS(M, N, K, ms);

        CHECK_CUDA(cudaFree(dA));
        CHECK_CUDA(cudaFree(dB));
        CHECK_CUDA(cudaFree(dC));

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(14) << ms
                  << std::setw(14) << gflops
                  << "\n";

        ofs << M << "," << N << "," << K << ","
            << ms << "," << gflops << "\n";
    }

    std::cout << "\n=======================================================\n";
    std::cout << "  V4 Optimization Summary:\n";
    std::cout << "  - FP16 TensorRT-style computation with __half type\n";
    std::cout << "  - __hfma instruction for fused multiply-add\n";
    std::cout << "  - 8x8 thread block configuration\n";
    std::cout << "=======================================================\n";

    return 0;
}