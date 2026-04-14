// GEMM V3: FastPath thread-level tiling
// - Aligned sizes use vectorized loads and branch-free writeback path.
// - Unaligned sizes fallback to generic boundary-checked kernel.

#include <cuda_runtime.h>
#include <mma.h>

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
constexpr int kBlockX = 8;
constexpr int kBlockY = 8;
constexpr int kTM = 4;  // default V3: 4x8
constexpr int kTN = 8;
constexpr int kBlockRows = kBlockY * kTM;
constexpr int kBlockCols = kBlockX * kTN;
}  // namespace

namespace large_tile {
constexpr int kTileK = 32;
constexpr int kBX = 16;
constexpr int kBY = 16;
constexpr int kTM = 8;
constexpr int kTN = 8;
constexpr int kCtaM = kBY * kTM;   // 128
constexpr int kCtaN = kBX * kTN;   // 128
}  // namespace large_tile

__global__ void GemmV3Kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K) {
    __shared__ float As[kBlockRows][kTileK + 1];
    __shared__ float Bs[kBlockCols][kTileK + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_start = blockIdx.y * kBlockRows + ty * kTM;
    int col_start = blockIdx.x * kBlockCols + tx * kTN;

    float sum[kTM][kTN] = {};

    const int tiles = (K + kTileK - 1) / kTileK;
    const int tid = ty * blockDim.x + tx;
    auto load_tile = [&](int tile_idx) {
        const int k_start = tile_idx * kTileK;
        for (int idx = tid; idx < kTileK * kBlockRows; idx += blockDim.x * blockDim.y) {
            const int k_idx = idx / kBlockRows;
            const int row_idx = idx % kBlockRows;
            const int global_row = blockIdx.y * kBlockRows + row_idx;
            const int global_col = k_start + k_idx;
            As[row_idx][k_idx] = (global_row < M && global_col < K)
                                     ? __ldg(A + global_row * K + global_col)
                                     : 0.0f;
        }

        for (int idx = tid; idx < kTileK * kBlockCols; idx += blockDim.x * blockDim.y) {
            const int k_idx = idx / kBlockCols;
            const int col_idx = idx % kBlockCols;
            const int global_row = k_start + k_idx;
            const int global_col = blockIdx.x * kBlockCols + col_idx;
            Bs[col_idx][k_idx] = (global_row < K && global_col < N)
                                     ? __ldg(B + global_row * N + global_col)
                                     : 0.0f;
        }
    };

    for (int t = 0; t < tiles; ++t) {
        load_tile(t);
        __syncthreads();

        #pragma unroll 4
        for (int k = 0; k < kTileK; ++k) {
            float b[kTN];
            #pragma unroll
            for (int j = 0; j < kTN; ++j) b[j] = Bs[tx * kTN + j][k];

            #pragma unroll
            for (int i = 0; i < kTM; ++i) {
                const float a = As[ty * kTM + i][k];
                #pragma unroll
                for (int j = 0; j < kTN; ++j) sum[i][j] += a * b[j];
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

__global__ void GemmV3KernelFast(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
    __shared__ float As[kBlockRows][kTileK + 1];
    __shared__ float Bs[kBlockCols][kTileK + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int row_start = blockIdx.y * kBlockRows + ty * kTM;
    const int col_start = blockIdx.x * kBlockCols + tx * kTN;

    float sum[kTM][kTN] = {};

    const int tiles = K / kTileK;
    for (int t = 0; t < tiles; ++t) {
        const int k_start = t * kTileK;

        for (int idx = tid; idx < kBlockRows * (kTileK / 4); idx += blockDim.x * blockDim.y) {
            const int row_idx = idx / (kTileK / 4);
            const int k4 = idx % (kTileK / 4);
            const int global_row = blockIdx.y * kBlockRows + row_idx;
            const float4 v = reinterpret_cast<const float4*>(
                                 A + global_row * K + k_start)[k4];
            As[row_idx][k4 * 4 + 0] = v.x;
            As[row_idx][k4 * 4 + 1] = v.y;
            As[row_idx][k4 * 4 + 2] = v.z;
            As[row_idx][k4 * 4 + 3] = v.w;
        }

        for (int idx = tid; idx < kTileK * (kBlockCols / 4); idx += blockDim.x * blockDim.y) {
            const int k_idx = idx / (kBlockCols / 4);
            const int c4 = idx % (kBlockCols / 4);
            const int global_row = k_start + k_idx;
            const int global_col = blockIdx.x * kBlockCols + c4 * 4;
            const float4 v = reinterpret_cast<const float4*>(
                                 B + global_row * N + global_col)[0];
            Bs[c4 * 4 + 0][k_idx] = v.x;
            Bs[c4 * 4 + 1][k_idx] = v.y;
            Bs[c4 * 4 + 2][k_idx] = v.z;
            Bs[c4 * 4 + 3][k_idx] = v.w;
        }

        __syncthreads();

        #pragma unroll 4
        for (int k = 0; k < kTileK; ++k) {
            float b[kTN];
            #pragma unroll
            for (int j = 0; j < kTN; ++j) b[j] = Bs[tx * kTN + j][k];

            #pragma unroll
            for (int i = 0; i < kTM; ++i) {
                const float a = As[ty * kTM + i][k];
                #pragma unroll
                for (int j = 0; j < kTN; ++j) sum[i][j] += a * b[j];
            }
        }
        __syncthreads();
    }

    // FastPath only runs for aligned sizes, so vectorized store is safe.
    #pragma unroll
    for (int i = 0; i < kTM; ++i) {
        const int r = row_start + i;
        float* row_ptr = C + r * N + col_start;
        #pragma unroll
        for (int j = 0; j < kTN; j += 4) {
            reinterpret_cast<float4*>(row_ptr + j)[0] =
                make_float4(sum[i][j + 0], sum[i][j + 1], sum[i][j + 2], sum[i][j + 3]);
        }
    }
}

// P0+P1 optimized: 128×128 CTA tile, 16×16 block (256 threads), 8×8 per thread, K-tile=32.
// Compute-to-load ratio: 64 FMA per loaded float; sync frequency 4x lower vs K-tile=8.
__global__ void __launch_bounds__(256, 2)
GemmV3LargeTileFast(const float* __restrict__ A,
                    const float* __restrict__ B,
                    float* __restrict__ C,
                    int M, int N, int K) {
    constexpr int LTK = large_tile::kTileK;
    constexpr int LTM = large_tile::kTM;
    constexpr int LTN = large_tile::kTN;
    constexpr int CM  = large_tile::kCtaM;
    constexpr int CN  = large_tile::kCtaN;
    constexpr int BX  = large_tile::kBX;

    __shared__ float As[CM][LTK + 1];
    __shared__ float Bs[CN][LTK + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BX + tx;
    const int row_start = blockIdx.y * CM + ty * LTM;
    const int col_start = blockIdx.x * CN + tx * LTN;

    float sum[LTM][LTN] = {};
    const int tiles = K / LTK;

    for (int t = 0; t < tiles; ++t) {
        const int k_start = t * LTK;

        // A tile: 128×16 = 2048 floats → 512 float4 loads (2 per thread)
        for (int idx = tid; idx < CM * (LTK / 4); idx += BX * large_tile::kBY) {
            const int row_idx = idx / (LTK / 4);
            const int k4 = idx % (LTK / 4);
            const int global_row = blockIdx.y * CM + row_idx;
            const float4 v = reinterpret_cast<const float4*>(
                                 A + global_row * K + k_start)[k4];
            As[row_idx][k4 * 4 + 0] = v.x;
            As[row_idx][k4 * 4 + 1] = v.y;
            As[row_idx][k4 * 4 + 2] = v.z;
            As[row_idx][k4 * 4 + 3] = v.w;
        }

        // B tile: 16×128 = 2048 floats → 512 float4 loads (2 per thread)
        for (int idx = tid; idx < LTK * (CN / 4); idx += BX * large_tile::kBY) {
            const int k_idx = idx / (CN / 4);
            const int c4 = idx % (CN / 4);
            const int global_row = k_start + k_idx;
            const int global_col = blockIdx.x * CN + c4 * 4;
            const float4 v = reinterpret_cast<const float4*>(
                                 B + global_row * N + global_col)[0];
            Bs[c4 * 4 + 0][k_idx] = v.x;
            Bs[c4 * 4 + 1][k_idx] = v.y;
            Bs[c4 * 4 + 2][k_idx] = v.z;
            Bs[c4 * 4 + 3][k_idx] = v.w;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < LTK; ++k) {
            float a[LTM], b[LTN];
            #pragma unroll
            for (int i = 0; i < LTM; ++i) a[i] = As[ty * LTM + i][k];
            #pragma unroll
            for (int j = 0; j < LTN; ++j) b[j] = Bs[tx * LTN + j][k];
            #pragma unroll
            for (int i = 0; i < LTM; ++i) {
                #pragma unroll
                for (int j = 0; j < LTN; ++j) {
                    sum[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < LTM; ++i) {
        const int r = row_start + i;
        float* row_ptr = C + r * N + col_start;
        #pragma unroll
        for (int j = 0; j < LTN; j += 4) {
            reinterpret_cast<float4*>(row_ptr + j)[0] =
                make_float4(sum[i][j + 0], sum[i][j + 1], sum[i][j + 2], sum[i][j + 3]);
        }
    }
}

// V6: V5 (128×128, kTileK=32) + register-level prefetch.
// Overlaps global memory loads with shared-memory compute by staging
// next K-tile data in registers before writing to shared memory.
__global__ void __launch_bounds__(256, 2)
GemmV6Kernel(const float* __restrict__ A,
             const float* __restrict__ B,
             float* __restrict__ C,
             int M, int N, int K) {
    constexpr int LTK = large_tile::kTileK;
    constexpr int LTM = large_tile::kTM;
    constexpr int LTN = large_tile::kTN;
    constexpr int CM  = large_tile::kCtaM;
    constexpr int CN  = large_tile::kCtaN;
    constexpr int BX  = large_tile::kBX;
    constexpr int BY  = large_tile::kBY;
    constexpr int NTHREADS = BX * BY;
    constexpr int A_LOADS = (CM * (LTK / 4) + NTHREADS - 1) / NTHREADS;
    constexpr int B_LOADS = (LTK * (CN / 4) + NTHREADS - 1) / NTHREADS;

    __shared__ float As[CM][LTK + 1];
    __shared__ float Bs[CN][LTK + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BX + tx;
    const int row_start = blockIdx.y * CM + ty * LTM;
    const int col_start = blockIdx.x * CN + tx * LTN;

    float sum[LTM][LTN] = {};
    const int tiles = K / LTK;

    float4 regA[A_LOADS];
    float4 regB[B_LOADS];
    int regA_idx[A_LOADS];
    int regB_idx[B_LOADS];

    // Load first tile into registers
    for (int li = 0; li < A_LOADS; ++li) {
        const int idx = tid + li * NTHREADS;
        if (idx < CM * (LTK / 4)) {
            const int row_idx = idx / (LTK / 4);
            const int k4 = idx % (LTK / 4);
            regA[li] = reinterpret_cast<const float4*>(
                           A + (blockIdx.y * CM + row_idx) * K)[k4];
            regA_idx[li] = idx;
        }
    }
    for (int li = 0; li < B_LOADS; ++li) {
        const int idx = tid + li * NTHREADS;
        if (idx < LTK * (CN / 4)) {
            const int k_idx = idx / (CN / 4);
            const int c4 = idx % (CN / 4);
            regB[li] = reinterpret_cast<const float4*>(
                           B + k_idx * N + blockIdx.x * CN + c4 * 4)[0];
            regB_idx[li] = idx;
        }
    }

    for (int t = 0; t < tiles; ++t) {
        // Write registers → shared memory
        for (int li = 0; li < A_LOADS; ++li) {
            const int idx = regA_idx[li];
            if (idx < CM * (LTK / 4)) {
                const int row_idx = idx / (LTK / 4);
                const int k4 = idx % (LTK / 4);
                As[row_idx][k4 * 4 + 0] = regA[li].x;
                As[row_idx][k4 * 4 + 1] = regA[li].y;
                As[row_idx][k4 * 4 + 2] = regA[li].z;
                As[row_idx][k4 * 4 + 3] = regA[li].w;
            }
        }
        for (int li = 0; li < B_LOADS; ++li) {
            const int idx = regB_idx[li];
            if (idx < LTK * (CN / 4)) {
                const int k_idx = idx / (CN / 4);
                const int c4 = idx % (CN / 4);
                Bs[c4 * 4 + 0][k_idx] = regB[li].x;
                Bs[c4 * 4 + 1][k_idx] = regB[li].y;
                Bs[c4 * 4 + 2][k_idx] = regB[li].z;
                Bs[c4 * 4 + 3][k_idx] = regB[li].w;
            }
        }
        __syncthreads();

        // Prefetch next tile into registers (overlaps with compute below)
        if (t + 1 < tiles) {
            const int next_k = (t + 1) * LTK;
            for (int li = 0; li < A_LOADS; ++li) {
                const int idx = tid + li * NTHREADS;
                if (idx < CM * (LTK / 4)) {
                    const int row_idx = idx / (LTK / 4);
                    const int k4 = idx % (LTK / 4);
                    regA[li] = reinterpret_cast<const float4*>(
                                   A + (blockIdx.y * CM + row_idx) * K + next_k)[k4];
                }
            }
            for (int li = 0; li < B_LOADS; ++li) {
                const int idx = tid + li * NTHREADS;
                if (idx < LTK * (CN / 4)) {
                    const int k_idx = idx / (CN / 4);
                    const int c4 = idx % (CN / 4);
                    regB[li] = reinterpret_cast<const float4*>(
                                   B + (next_k + k_idx) * N + blockIdx.x * CN + c4 * 4)[0];
                }
            }
        }

        // Compute with current tile in shared memory
        #pragma unroll
        for (int k = 0; k < LTK; ++k) {
            float a[LTM], b[LTN];
            #pragma unroll
            for (int i = 0; i < LTM; ++i) a[i] = As[ty * LTM + i][k];
            #pragma unroll
            for (int j = 0; j < LTN; ++j) b[j] = Bs[tx * LTN + j][k];
            #pragma unroll
            for (int i = 0; i < LTM; ++i) {
                #pragma unroll
                for (int j = 0; j < LTN; ++j) {
                    sum[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < LTM; ++i) {
        const int r = row_start + i;
        float* row_ptr = C + r * N + col_start;
        #pragma unroll
        for (int j = 0; j < LTN; j += 4) {
            reinterpret_cast<float4*>(row_ptr + j)[0] =
                make_float4(sum[i][j + 0], sum[i][j + 1], sum[i][j + 2], sum[i][j + 3]);
        }
    }
}

__global__ void GemmV3KernelTensorCore(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int N, int K) {
    if (threadIdx.x >= 32) return;

    const int row_start = blockIdx.y * 16;
    const int col_start = blockIdx.x * 16;
    if (row_start + 15 >= M || col_start + 15 >= N) return;

    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fragB;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> fragC;
    wmma::fill_fragment(fragC, 0.0f);

    const int k_tiles = K / 8;
    for (int kt = 0; kt < k_tiles; ++kt) {
        const int k0 = kt * 8;
        wmma::load_matrix_sync(fragA, A + row_start * K + k0, K);
        wmma::load_matrix_sync(fragB, B + k0 * N + col_start, N);
        wmma::mma_sync(fragC, fragA, fragB, fragC);
    }

    wmma::store_matrix_sync(C + row_start * N + col_start, fragC, N, wmma::mem_row_major);
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
    constexpr int kMaxGpuRunDim = 4096;
    constexpr int kMaxCpuVerifyDim = 1024;
    constexpr int kTileRows = kBlockRows;
    constexpr int kTileCols = kBlockCols;
    auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/gemm_v3_results.csv");
    ofs << "id,M,N,K,gpu_ms,gflops,max_abs_diff,check\n";

    for (size_t i = 0; i < cases.size(); ++i) {
        int M = cases[i].rows, N = cases[i].cols, K = (M + N) / 2;
        const bool do_gpu_run = (M <= kMaxGpuRunDim && N <= kMaxGpuRunDim && K <= kMaxGpuRunDim);
        std::vector<float> A(static_cast<size_t>(M) * K),
            B(static_cast<size_t>(K) * N),
            cpu(static_cast<size_t>(M) * N),
            gpu(static_cast<size_t>(M) * N);
        common::InitMatrix(A, M, K);
        common::InitMatrix(B, K, N);
        const bool do_cpu_verify = (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim && K <= kMaxCpuVerifyDim);
        if (do_cpu_verify) {
            GemmCPU(A.data(), B.data(), cpu.data(), M, N, K);
        }

        float gpu_ms = 0.0f;
        if (do_gpu_run) {
            float *dA, *dB, *dC;
            CHECK_CUDA(cudaMalloc(&dA, A.size() * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&dB, B.size() * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&dC, cpu.size() * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

            dim3 block_lt(large_tile::kBX, large_tile::kBY);
            dim3 grid_lt((N + large_tile::kCtaN - 1) / large_tile::kCtaN,
                         (M + large_tile::kCtaM - 1) / large_tile::kCtaM);
            dim3 block_tt(kBlockX, kBlockY);
            dim3 grid_tt((N + kTileCols - 1) / kTileCols, (M + kTileRows - 1) / kTileRows);

            const bool v6_path = (M >= 512 && N >= 512 &&
                                  (M % large_tile::kCtaM) == 0 &&
                                  (N % large_tile::kCtaN) == 0 &&
                                  (K % large_tile::kTileK) == 0);
            const bool lt_fast = (M >= 512 && N >= 512 &&
                                  (M % large_tile::kCtaM) == 0 &&
                                  (N % large_tile::kCtaN) == 0 &&
                                  (K % large_tile::kTileK) == 0);
            const bool fast_path = ((M % kTileRows) == 0 && (N % kTileCols) == 0 && (K % kTileK) == 0);

            auto run_kernel = [&]() {
                if (v6_path) {
                    GemmV6Kernel<<<grid_lt, block_lt>>>(dA, dB, dC, M, N, K);
                } else if (lt_fast) {
                    GemmV3LargeTileFast<<<grid_lt, block_lt>>>(dA, dB, dC, M, N, K);
                } else if (fast_path) {
                    GemmV3KernelFast<<<grid_tt, block_tt>>>(dA, dB, dC, M, N, K);
                } else {
                    GemmV3Kernel<<<grid_tt, block_tt>>>(dA, dB, dC, M, N, K);
                }
            };

            run_kernel();
            CHECK_CUDA(cudaDeviceSynchronize());

            cudaEvent_t s, e;
            CHECK_CUDA(cudaEventCreate(&s));
            CHECK_CUDA(cudaEventCreate(&e));
            std::vector<float> gpu_times;
            gpu_times.reserve(kRepeat);
            for (int rep = 0; rep < kRepeat; ++rep) {
                CHECK_CUDA(cudaEventRecord(s));
                run_kernel();
                CHECK_CUDA(cudaEventRecord(e));
                CHECK_CUDA(cudaEventSynchronize(e));
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

            CHECK_CUDA(cudaMemcpy(gpu.data(), dC, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
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
            const float tol = 1e-3f;
            ok = common::CheckEqual(cpu, gpu, tol);
            max_abs_diff = common::MaxAbsDiff(cpu, gpu);
            check = ok ? "PASS" : "FAIL";
        }
        double gflops = (gpu_ms > 0.0f) ? (2.0 * M * N * K / (gpu_ms * 1e6)) : 0.0;

        std::cout << "M=" << M << " N=" << N << " K=" << K
                  << " | " << std::fixed << std::setprecision(3) << gpu_ms << " ms"
                  << " | " << std::setprecision(1) << gflops << " GFLOPS"
                  << " | " << check << "\n";

        ofs << i << "," << M << "," << N << "," << K << ","
            << gpu_ms << "," << gflops << ","
            << max_abs_diff << "," << check << "\n";
    }
    return 0;
}
