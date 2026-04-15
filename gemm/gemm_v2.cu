// GEMM V2: Thread-level register tiling (optimization plan b)
//
// Optimization over V1:
//   b — Each thread computes a TM×TN (4×4) sub-block using register accumulators.
//       One SMEM load of `a` feeds TN=4 FMAs, one load of `b` feeds TM=4 FMAs.
//       Arithmetic intensity: 16 FMA per (TM+TN)=8 SMEM loads = 2 FMA/load,
//       vs V1's 1 FMA per 2 loads = 4× improvement in SMEM utilization.

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
constexpr int kBlockM = 32;
constexpr int kBlockN = 32;
constexpr int kTileK = 8;
constexpr int kTM = 4;
constexpr int kTN = 4;
constexpr int kBX = kBlockN / kTN;   // 8  threads along N
constexpr int kBY = kBlockM / kTM;   // 8  threads along M
}  // namespace

__global__ void GemmV2Kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {
    // As: standard layout [row][k], Bs: transposed layout [col][k]
    __shared__ float As[kBlockM][kTileK + 1];
    __shared__ float Bs[kBlockN][kTileK + 1];

    const int tx = threadIdx.x;   // 0..7
    const int ty = threadIdx.y;   // 0..7
    const int tid = ty * kBX + tx;
    const int row_start = blockIdx.y * kBlockM + ty * kTM;
    const int col_start = blockIdx.x * kBlockN + tx * kTN;

    const bool can_vec_a = (K % 4 == 0);
    const bool can_vec_b = (N % 4 == 0);

    float sum[kTM][kTN] = {};

    const int k_tiles = (K + kTileK - 1) / kTileK;

    for (int t = 0; t < k_tiles; ++t) {
        const int k0 = t * kTileK;

        // ---- Load A tile: 32×8 = 256 floats = 64 float4, all 64 threads ----
        {
            const int a_row = tid / (kTileK / 4);    // 0..31
            const int a_k4  = tid % (kTileK / 4);    // 0..1
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

        // ---- Load B tile: 8×32 = 256 floats = 64 float4, transposed store ----
        {
            const int b_k  = tid / (kBlockN / 4);    // 0..7
            const int b_n4 = tid % (kBlockN / 4);    // 0..7
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

        // ---- Compute: each thread accumulates TM×TN sub-block ----
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

    // ---- Write back with boundary checks ----
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

            dim3 block(kBX, kBY);
            dim3 grid((N + kBlockN - 1) / kBlockN, (M + kBlockM - 1) / kBlockM);

            GemmV2Kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
            CHECK_CUDA(cudaDeviceSynchronize());

            cudaEvent_t s, e;
            CHECK_CUDA(cudaEventCreate(&s));
            CHECK_CUDA(cudaEventCreate(&e));
            std::vector<float> gpu_times;
            gpu_times.reserve(kRepeat);
            for (int rep = 0; rep < kRepeat; ++rep) {
                CHECK_CUDA(cudaEventRecord(s));
                GemmV2Kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
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
