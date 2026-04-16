// GEMM V3: 寄存器预取隐藏延迟
//
// 相对于 V2 的优化:
// — 软件流水线: 当从共享内存计算当前 K-tile 时, 预取下一个 K-tile 数据到寄存器
//   全局内存延迟与共享内存计算重叠, 减少 Stall Long Scoreboard
//
// 同时增大 tile 大小 (64×64, TM=TN=8, K-tile=8) 以提高计算/内存比
// 要求对齐维度 (M%64==0, N%64==0, K%8==0)

#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

namespace {
constexpr int kTileK = 8;
constexpr int kBX = 8;
constexpr int kBY = 8;
constexpr int kTM = 8;
constexpr int kTN = 8;
constexpr int kCtaM = kBY * kTM;
constexpr int kCtaN = kBX * kTN;
constexpr int kThreads = kBX * kBY;
}  // namespace

__global__ void __launch_bounds__(256, 2)
GemmV3Kernel(const float* __restrict__ A,
             const float* __restrict__ B,
             float* __restrict__ C,
             int M, int N, int K) {
    constexpr int A_LOADS = (kCtaM * (kTileK / 4)) / kThreads;
    constexpr int B_LOADS = (kTileK * (kCtaN / 4)) / kThreads;

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

    int a_row_base = blockIdx.y * kCtaM;
    int b_col_base = blockIdx.x * kCtaN;

    auto loadA = [&](int k_base) {
        #pragma unroll
        for (int i = 0; i < A_LOADS; ++i) {
            int idx = tid + i * kThreads;
            int row = idx / (kTileK / 4);
            int k4 = idx % (kTileK / 4);
            int gk = k_base + k4;
            regA[i] = __ldg(reinterpret_cast<const float4*>(A + (a_row_base + row) * K + gk)[0]);
        }
    };

    auto loadB = [&](int k_base) {
        #pragma unroll
        for (int i = 0; i < B_LOADS; ++i) {
            int idx = tid + i * kThreads;
            int k = idx / (kCtaN / 4);
            int col4 = idx % (kCtaN / 4);
            int gk = k_base + k;
            regB[i] = __ldg(reinterpret_cast<const float4*>(B + gk * N + b_col_base + col4 * 4)[0]);
        }
    };

    auto storeA = [&]() {
        #pragma unroll
        for (int i = 0; i < A_LOADS; ++i) {
            int idx = tid + i * kThreads;
            int row = idx / (kTileK / 4);
            int k4 = idx % (kTileK / 4);
            float4 v = regA[i];
            As[row][k4 * 4 + 0] = v.x;
            As[row][k4 * 4 + 1] = v.y;
            As[row][k4 * 4 + 2] = v.z;
            As[row][k4 * 4 + 3] = v.w;
        }
    };

    auto storeB = [&]() {
        #pragma unroll
        for (int i = 0; i < B_LOADS; ++i) {
            int idx = tid + i * kThreads;
            int k = idx / (kCtaN / 4);
            int col4 = idx % (kCtaN / 4);
            float4 v = regB[i];
            Bs[col4 * 4 + 0][k] = v.x;
            Bs[col4 * 4 + 1][k] = v.y;
            Bs[col4 * 4 + 2][k] = v.z;
            Bs[col4 * 4 + 3][k] = v.w;
        }
    };

    loadA(0);
    loadB(0);

    for (int t = 0; t < tiles; ++t) {
        storeA();
        storeB();
        __syncthreads();

        if (t + 1 < tiles) {
            int next_k = (t + 1) * kTileK;
            loadA(next_k);
            loadB(next_k);
        }

        for (int kk = 0; kk < kTileK; ++kk) {
            float a_reg[kTM];
            float b_reg[kTN];
            #pragma unroll
            for (int i = 0; i < kTM; ++i) a_reg[i] = As[ty * kTM + i][kk];
            #pragma unroll
            for (int j = 0; j < kTN; ++j) b_reg[j] = Bs[tx * kTN + j][kk];
            #pragma unroll
            for (int i = 0; i < kTM; ++i) {
                #pragma unroll
                for (int j = 0; j < kTN; ++j) {
                    sum[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < kTM; ++i) {
        float* c_row = C + (row_start + i) * N + col_start;
        #pragma unroll
        for (int j = 0; j < kTN; j += 4) {
            reinterpret_cast<float4*>(c_row + j)[0] =
                make_float4(sum[i][j], sum[i][j + 1], sum[i][j + 2], sum[i][j + 3]);
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
    constexpr int kMaxCpuVerifyDim = 1024;
    constexpr int kMaxGpuRunDim = 4096;
    auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/gemm_v3_results.csv");
    ofs << "id,group,M,N,K,gpu_ms,gflops,max_abs_diff,check\n";

    for (size_t i = 0; i < cases.size(); ++i) {
        int M = cases[i].rows, N = cases[i].cols, K = M;
        const bool do_gpu_run = (M <= kMaxGpuRunDim && N <= kMaxGpuRunDim && K <= kMaxGpuRunDim);
        const bool aligned = (M % kCtaM == 0) && (N % kCtaN == 0) && (K % kTileK == 0);
        std::vector<float> A(static_cast<size_t>(M) * K),
            B(static_cast<size_t>(K) * N),
            cpu(static_cast<size_t>(M) * N),
            gpu(static_cast<size_t>(M) * N);
        common::InitMatrix(A, M, K);
        common::InitMatrix(B, K, N);
        GemmCPU(A.data(), B.data(), cpu.data(), M, N, K);

        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(kBX, kBY);
        dim3 grid((N + kCtaN - 1) / kCtaN, (M + kCtaM - 1) / kCtaM);
        GemmV3Kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        for (int r = 0; r < kRepeat; ++r) {
            GemmV3Kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        ms /= kRepeat;

        CHECK_CUDA(cudaMemcpy(gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = common::CheckEqual(cpu, gpu, 1e-3f);
        double gflops = (2.0 * M * N * K) / (ms * 1e6);

        std::cout << M << "x" << N << "x" << K << " | " << std::fixed << std::setprecision(4) << ms << " ms"
                  << " | " << std::setprecision(1) << gflops << " GFLOP/s"
                  << " | " << (ok ? "PASS" : "FAIL") << "\n";

        ofs << i << ",gemm_v3," << M << "," << N << "," << K << ","
            << ms << "," << gflops << ","
            << common::MaxAbsDiff(cpu, gpu) << ","
            << (ok ? "PASS" : "FAIL") << "\n";

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
    }
    return 0;
}