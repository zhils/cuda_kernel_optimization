// GEMM V2: 线程级寄存器分块
//
// 相对于 V1 的优化:
// — 每个线程计算 TM×TN (4×4) 子块，使用寄存器累加器
// — 一次共享内存加载的 `a` 供给 TN=4 个 FMA，一次加载的 `b` 供给 TM=4 个 FMA
// — 算术强度: 16 FMA / (TM+TN)=8 共享内存加载 = 2 FMA/加载
// — 相比 V1 的 1 FMA / 2 加载 = 4× 共享内存利用率提升

#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kTileK = 8;
constexpr int kTM = 8;
constexpr int kTN = 8;
constexpr int kBX = kBlockN / kTN;
constexpr int kBY = kBlockM / kTM;

__global__ void GemmV2Kernel(const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[kBlockM][kTileK + 1];
    __shared__ float Bs[kTileK][kBlockN + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kBX + tx;
    const int row_start = blockIdx.y * kBlockM + ty * kTM;
    const int col_start = blockIdx.x * kBlockN + tx * kTN;

    float sum[kTM][kTN] = {};

    const int k_tiles = (K + kTileK - 1) / kTileK;

    for (int t = 0; t < k_tiles; ++t) {
        const int k0 = t * kTileK;

        const int r = tid / (kTileK / 4);
        const int k = tid % (kTileK / 4);
        const int g_r = blockIdx.y * kBlockM + r;
        const int g_k = k0 + k * 4;

        const float4 v_a = __ldg(reinterpret_cast<const float4*>(A + g_r * K + g_k));
        As[r][k * 4 + 0] = v_a.x;
        As[r][k * 4 + 1] = v_a.y;
        As[r][k * 4 + 2] = v_a.z;
        As[r][k * 4 + 3] = v_a.w;

        const int k_b = tid / (kBlockN / 4);
        const int c = tid % (kBlockN / 4);
        const int g_r_b = k0 + k_b;
        const int g_c = blockIdx.x * kBlockN + c * 4;

        const float4 v_b = __ldg(reinterpret_cast<const float4*>(B + g_r_b * N + g_c));
        Bs[c * 4 + 0][k_b] = v_b.x;
        Bs[c * 4 + 1][k_b] = v_b.y;
        Bs[c * 4 + 2][k_b] = v_b.z;
        Bs[c * 4 + 3][k_b] = v_b.w;

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
        #pragma unroll
        for (int j = 0; j < kTN; ++j) {
            const int g_r = row_start + i;
            const int g_c = col_start + j;
            if (g_r < M && g_c < N) {
                C[g_r * N + g_c] = sum[i][j];
            }
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
    std::ofstream ofs("data/results/gemm_v2_results.csv");
    ofs << "id,group,M,N,K,gpu_ms,gflops,max_abs_diff,check\n";

    for (size_t i = 0; i < cases.size(); ++i) {
        int M = cases[i].rows, N = cases[i].cols, K = M;
        int n = M * N;
        std::vector<float> A(n), B(n), C_cpu(n), C_gpu(n);
        common::InitMatrix(A, M, K);
        common::InitMatrix(B, K, N);
        GemmCPU(A.data(), B.data(), C_cpu.data(), M, N, K);

        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(kBX, kBY);
        dim3 grid((N + kBlockN - 1) / kBlockN, (M + kBlockM - 1) / kBlockM);
        GemmV2Kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        for (int r = 0; r < kRepeat; ++r) {
            GemmV2Kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        ms /= kRepeat;

        CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = common::CheckEqual(C_cpu, C_gpu, 1e-3f);
        double gflops = (2.0 * M * N * K) / (ms * 1e6);

        std::cout << M << "x" << N << "x" << K << " | " << std::fixed << std::setprecision(4) << ms << " ms"
                  << " | " << std::setprecision(1) << gflops << " GFLOP/s"
                  << " | " << (ok ? "PASS" : "FAIL") << "\n";

        ofs << i << ",gemm_v2," << M << "," << N << "," << K << ","
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