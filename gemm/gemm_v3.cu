// GEMM V3: Register prefetch for latency hiding (optimization plan c)
//
// Optimization over V2:
//   c — Software pipelining: prefetch next K-tile data into registers while
//       computing current K-tile from shared memory. Global memory latency
//       is overlapped with shared memory compute, reducing Stall Long Scoreboard.
//
// Also scales up tile size (128×128, TM=TN=8, K-tile=32) for higher
// compute-to-memory ratio: 64 FMA per SMEM load pair vs V2's 16.
// Requires aligned dimensions (M%128==0, N%128==0, K%32==0).

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
constexpr int kTileK = 32;
constexpr int kBX = 16;
constexpr int kBY = 16;
constexpr int kTM = 8;
constexpr int kTN = 8;
constexpr int kCtaM = kBY * kTM;   // 128
constexpr int kCtaN = kBX * kTN;   // 128
}  // namespace

__global__ void __launch_bounds__(256, 2)
GemmV3Kernel(const float* __restrict__ A,
             const float* __restrict__ B,
             float* __restrict__ C,
             int M, int N, int K) {
    constexpr int NTHREADS = kBX * kBY;
    constexpr int A_LOADS = (kCtaM * (kTileK / 4)) / NTHREADS;   // 4
    constexpr int B_LOADS = (kTileK * (kCtaN / 4)) / NTHREADS;   // 4

    __shared__ float As[kCtaM][kTileK + 1];
    __shared__ float Bs[kCtaN][kTileK + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kBX + tx;
    const int row_start = blockIdx.y * kCtaM + ty * kTM;
    const int col_start = blockIdx.x * kCtaN + tx * kTN;

    float sum[kTM][kTN] = {};
    const int tiles = K / kTileK;

    // ---- Register buffers for prefetch ----
    float4 regA[A_LOADS];
    float4 regB[B_LOADS];

    // ---- Prefetch first K-tile into registers ----
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
        // ---- Write prefetched registers → shared memory ----
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

        // ---- Prefetch NEXT K-tile into registers (overlaps with compute) ----
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

        // ---- Compute with current shared memory tile ----
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

    // ---- Vectorized write-back (float4, no boundary checks) ----
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
        const bool do_cpu_verify =
            (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim && K <= kMaxCpuVerifyDim);
        if (do_cpu_verify) {
            GemmCPU(A.data(), B.data(), cpu.data(), M, N, K);
        }

        float gpu_ms = 0.0f;
        if (do_gpu_run && aligned) {
            float *dA, *dB, *dC;
            CHECK_CUDA(cudaMalloc(&dA, A.size() * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&dB, B.size() * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&dC, gpu.size() * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

            dim3 block(kBX, kBY);
            dim3 grid((N + kCtaN - 1) / kCtaN, (M + kCtaM - 1) / kCtaM);

            GemmV3Kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
            CHECK_CUDA(cudaDeviceSynchronize());

            cudaEvent_t s, e;
            CHECK_CUDA(cudaEventCreate(&s));
            CHECK_CUDA(cudaEventCreate(&e));
            std::vector<float> gpu_times;
            gpu_times.reserve(kRepeat);
            for (int rep = 0; rep < kRepeat; ++rep) {
                CHECK_CUDA(cudaEventRecord(s));
                GemmV3Kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
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
        } else if (!aligned) {
            check = "SKIP_UNALIGNED";
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
