// GEMM V1: 共享内存分块 + float4 向量化加载 + __ldg + Bank Conflict 避免
//
// 相对于 V0 的优化:
// — 共享内存分块: A/B 元素每个 tile 只从全局内存加载一次, 全局流量从 O(M*N*K) 减少到 M*K + K*N + M*N
// — float4 协作加载 (减少 4x 加载指令), __ldg 只读缓存,
// — 共享内存 +1 padding 避免 bank conflict

#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

#define kTile 32
#define kTileK 32

__global__ void GemmV1Kernel(const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {
        
    __shared__ float As[kTile][kTileK + 1];
    __shared__ float Bs[kTileK][kTile + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kTile + tx;
    const int row = blockIdx.y * kTile + ty;
    const int col = blockIdx.x * kTile + tx;

    float sum = 0.0f;
    const int k_tiles = (K + kTileK - 1) / kTileK;

    for (int t = 0; t < k_tiles; ++t) {
        const int k0 = t * kTileK;

        constexpr int kALoads = kTile * (kTileK / 4);
        constexpr int kBLoads = kTileK * (kTile / 4);

        if (tid < kALoads) {
            const int r = tid / (kTileK / 4);
            const int k = tid % (kTileK / 4);
            const int g_r = blockIdx.y * kTile + r;
            const int g_k = k0 + k * 4;

            const float4 v_a = __ldg(reinterpret_cast<const float4*>(A + g_r * K + g_k));
            As[r][k * 4 + 0] = v_a.x;
            As[r][k * 4 + 1] = v_a.y;
            As[r][k * 4 + 2] = v_a.z;
            As[r][k * 4 + 3] = v_a.w;
        }

        if (tid < kBLoads) {
            const int k_b = tid / (kTile / 4);
            const int c = tid % (kTile / 4);
            const int g_r_b = k0 + k_b;
            const int g_c = blockIdx.x * kTile + c * 4;

            const float4 v_b = __ldg(reinterpret_cast<const float4*>(B + g_r_b * N + g_c));
            Bs[c * 4 + 0][k_b] = v_b.x;
            Bs[c * 4 + 1][k_b] = v_b.y;
            Bs[c * 4 + 2][k_b] = v_b.z;
            Bs[c * 4 + 3][k_b] = v_b.w;
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
        GemmCPU(A.data(), B.data(), C_cpu.data(), M, N, K);

        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(kTile, kTile);
        dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
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
        bool ok = common::CheckEqual(C_cpu, C_gpu, 1e-3f);
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