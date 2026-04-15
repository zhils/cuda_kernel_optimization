// GEMM V1: Shared memory tiling + float4 vectorized loading + __ldg + bank conflict padding
//
// Optimizations over V0 (optimization plan a.i and a.ii):
//   a.i  — Shared memory tiling: A/B elements loaded from global memory once per tile,
//           total global traffic reduced from O(M*N*K) to M*K + K*N + M*N
//   a.ii — float4 cooperative loading (4x fewer load instructions), __ldg read-only cache,
//           shared memory +1 padding for bank conflict avoidance

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
constexpr int kTile = 16;
constexpr int kTileK = 32;
}  // namespace

__global__ void GemmV1Kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {
    __shared__ float As[kTile][kTileK + 1];   // +1 padding avoids bank conflict
    __shared__ float Bs[kTileK][kTile + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kTile + tx;           // 0..255
    const int row = blockIdx.y * kTile + ty;
    const int col = blockIdx.x * kTile + tx;

    const bool can_vec_a = (K % 4 == 0);
    const bool can_vec_b = (N % 4 == 0);

    float sum = 0.0f;
    const int k_tiles = (K + kTileK - 1) / kTileK;

    for (int t = 0; t < k_tiles; ++t) {
        const int k0 = t * kTileK;

        // ---- Cooperative float4 loading ----
        // A tile: 16×32 = 512 floats = 128 float4  →  threads 0..127
        // B tile: 32×16 = 512 floats = 128 float4  →  threads 128..255
        // All 256 threads participate, each loading exactly one float4.

        if (tid < kTile * (kTileK / 4)) {
            const int a_row = tid / (kTileK / 4);          // 0..15
            const int a_k4  = tid % (kTileK / 4);          // 0..7
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
            const int b_tid = tid - kTile * (kTileK / 4);  // 0..127
            const int b_row = b_tid / (kTile / 4);         // 0..31
            const int b_n4  = b_tid % (kTile / 4);         // 0..3
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
        const bool do_gpu_run = (M <= kMaxGpuRunDim && N <= kMaxGpuRunDim && K <= kMaxGpuRunDim);
        std::vector<float> A(static_cast<size_t>(M) * K),
            B(static_cast<size_t>(K) * N),
            cpu(static_cast<size_t>(M) * N),
            gpu(static_cast<size_t>(M) * N);
        common::InitMatrix(A, M, K);
        common::InitMatrix(B, K, N);
        const bool do_cpu_verify =
            (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim && K <= kMaxCpuVerifyDim);
        if (do_cpu_verify) {
            GemmCPU(A.data(), B.data(), cpu.data(), M, N, K);
        }

        float gpu_ms = 0.0f;
        if (do_gpu_run) {
            float *dA, *dB, *dC;
            CHECK_CUDA(cudaMalloc(&dA, A.size() * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&dB, B.size() * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&dC, gpu.size() * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

            dim3 block(kTile, kTile);
            dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);

            GemmV1Kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
            CHECK_CUDA(cudaDeviceSynchronize());

            cudaEvent_t s, e;
            CHECK_CUDA(cudaEventCreate(&s));
            CHECK_CUDA(cudaEventCreate(&e));
            std::vector<float> gpu_times;
            gpu_times.reserve(kRepeat);
            for (int rep = 0; rep < kRepeat; ++rep) {
                CHECK_CUDA(cudaEventRecord(s));
                GemmV1Kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
                CHECK_CUDA(cudaEventRecord(e));
                CHECK_CUDA(cudaEventSynchronize(e));
                CHECK_CUDA(cudaGetLastError());
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

            CHECK_CUDA(cudaMemcpy(gpu.data(), dC, gpu.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
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

        ofs << cases[i].id << "," << cases[i].group << "," << M << "," << N << "," << K << ","
            << gpu_ms << "," << gflops << "," << max_abs_diff << "," << check << "\n";
    }
    return 0;
}
