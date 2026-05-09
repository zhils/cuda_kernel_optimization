#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

// ============================================================================
// Fused L2 Norm Q/K v2：提升计算强度
// ============================================================================
// 优化策略：
//   1. 每 block 处理 2 行（kRowsPerBlock=2），GFLOPs/Byte 翻倍
//   2. ILP（指令级并行）：每线程 4 寄存器累加，减少归约开销占比
//   3. __fmaf_rn 融合乘加，减少指令数
//   4. #pragma unroll 展开内层循环，暴露 ILP
//   5. Warp Shuffle 归约（继承 v1）
// ============================================================================

constexpr float kEps = 1e-6f;
constexpr int kRowsPerBlock = 2;
constexpr int kILP = 4;  // 每线程 4 路 ILP

// ----------------------------------------------------------------------------
// Fused L2 Norm Kernel v2：每 block 处理 2 行，ILP 展开
// 3D grid: blockIdx.x = b, blockIdx.y = n/kRowsPerBlock, blockIdx.z = 0(Q)/1(K)
// ----------------------------------------------------------------------------
__global__ void FusedL2NormV2Kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ Q_hat,
    float* __restrict__ K_hat,
    int B, int N_q, int H_q, int N_k, int H_k)
{
    int b = blockIdx.x;
    int group = blockIdx.y;
    int qk = blockIdx.z;
    if (b >= B) return;

    bool is_q = (qk == 0);
    int N = is_q ? N_q : N_k;
    int H = is_q ? H_q : H_k;
    int row_start = group * kRowsPerBlock;
    if (row_start >= N) return;
    int rows_this_block = min(kRowsPerBlock, N - row_start);

    const float* X = is_q ? Q : K;
    float* X_hat = is_q ? Q_hat : K_hat;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nthreads = blockDim.x;

    // 每线程 2 行 × kILP 个独立累加器
    float sum0[kILP] = {0.0f};
    float sum1[kILP] = {0.0f};
    float val0[kILP], val1[kILP];

    const float* row_ptr0 = X + (b * N + row_start + 0) * H;

    // ------------------------------------------------------------------
    // Phase 1: ILP 展开 + float4 隐式向量化 + __fmaf_rn 累加
    // 外层循环步长 = nthreads × kILP
    // ------------------------------------------------------------------
    {
        int h = tid * kILP;
        #pragma unroll 1  // 外层保持循环让编译器自由调度
        for (; h + kILP * nthreads <= H; h += kILP * nthreads) {
            #pragma unroll
            for (int i = 0; i < kILP; ++i) {
                val0[i] = row_ptr0[h + i];
                sum0[i] = __fmaf_rn(val0[i], val0[i], sum0[i]);
            }
        }
        // 尾部：处理剩余元素
        #pragma unroll
        for (int i = 0; i < kILP && h + i < H; ++i) {
            val0[i] = row_ptr0[h + i];
            sum0[i] = __fmaf_rn(val0[i], val0[i], sum0[i]);
        }
    }

    // 第 2 行
    if (rows_this_block > 1) {
        const float* row_ptr1 = X + (b * N + row_start + 1) * H;
        int h = tid * kILP;
        for (; h + kILP * nthreads <= H; h += kILP * nthreads) {
            #pragma unroll
            for (int i = 0; i < kILP; ++i) {
                val1[i] = row_ptr1[h + i];
                sum1[i] = __fmaf_rn(val1[i], val1[i], sum1[i]);
            }
        }
        #pragma unroll
        for (int i = 0; i < kILP && h + i < H; ++i) {
            val1[i] = row_ptr1[h + i];
            sum1[i] = __fmaf_rn(val1[i], val1[i], sum1[i]);
        }
    }

    // ------------------------------------------------------------------
    // Phase 2: 合并 kILP 路累加器
    // ------------------------------------------------------------------
    float local_sum0 = 0.0f;
    #pragma unroll
    for (int i = 0; i < kILP; ++i) {
        local_sum0 += sum0[i];
    }
    float local_sum1 = 0.0f;
    if (rows_this_block > 1) {
        #pragma unroll
        for (int i = 0; i < kILP; ++i) {
            local_sum1 += sum1[i];
        }
    }

    // ------------------------------------------------------------------
    // Phase 3: Warp Shuffle 归约（第 0 行）
    // ------------------------------------------------------------------
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum0 += __shfl_down_sync(0xffffffff, local_sum0, offset);
    }
    __shared__ float s_warp[32];
    if (lane == 0) s_warp[warp] = local_sum0;
    __syncthreads();

    if (warp == 0) {
        float val = (tid < ((nthreads + 31) / 32)) ? s_warp[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tid == 0) s_warp[0] = sqrtf(val + kEps);
    }
    __syncthreads();
    float inv_norm0 = 1.0f / s_warp[0];

    // 第 1 行归约
    float inv_norm1 = 0.0f;
    if (rows_this_block > 1) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum1 += __shfl_down_sync(0xffffffff, local_sum1, offset);
        }
        if (lane == 0) s_warp[warp] = local_sum1;
        __syncthreads();
        if (warp == 0) {
            float val = (tid < ((nthreads + 31) / 32)) ? s_warp[lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (tid == 0) s_warp[0] = sqrtf(val + kEps);
        }
        __syncthreads();
        inv_norm1 = 1.0f / s_warp[0];
    }

    // ------------------------------------------------------------------
    // Phase 4: 写回（第 2 次访存，直接向量化写）
    //         利用 4 路 ILP 掩盖写延迟
    // ------------------------------------------------------------------
    {
        float* out_ptr0 = X_hat + (b * N + row_start + 0) * H;
        int h = tid * kILP;
        for (; h + kILP * nthreads <= H; h += kILP * nthreads) {
            #pragma unroll
            for (int i = 0; i < kILP; ++i) {
                float v = row_ptr0[h + i];
                out_ptr0[h + i] = v * inv_norm0;
            }
        }
        #pragma unroll
        for (int i = 0; i < kILP && h + i < H; ++i) {
            float v = row_ptr0[h + i];
            out_ptr0[h + i] = v * inv_norm0;
        }
    }

    if (rows_this_block > 1) {
        float* out_ptr1 = X_hat + (b * N + row_start + 1) * H;
        const float* row_ptr1 = X + (b * N + row_start + 1) * H;
        int h = tid * kILP;
        for (; h + kILP * nthreads <= H; h += kILP * nthreads) {
            #pragma unroll
            for (int i = 0; i < kILP; ++i) {
                float v = row_ptr1[h + i];
                out_ptr1[h + i] = v * inv_norm1;
            }
        }
        #pragma unroll
        for (int i = 0; i < kILP && h + i < H; ++i) {
            float v = row_ptr1[h + i];
            out_ptr1[h + i] = v * inv_norm1;
        }
    }
}

// ============================================================================
// Wrapper
// ============================================================================
static void RunGpuV2(const float* d_Q, const float* d_K,
                     float* d_Q_hat, float* d_K_hat,
                     int B, int N_q, int H_q, int N_k, int H_k)
{
    dim3 block(256);
    int max_N = std::max(N_q, N_k);
    int groups = (max_N + kRowsPerBlock - 1) / kRowsPerBlock;
    dim3 grid(B, groups, 2);

    FusedL2NormV2Kernel<<<grid, block>>>(
        d_Q, d_K, d_Q_hat, d_K_hat,
        B, N_q, H_q, N_k, H_k);
}

// ============================================================================
// CPU Reference — 同 v0
// ============================================================================
static void L2Norm_CPU(const float* X, float* X_hat, int B, int N, int H) {
    for (int b = 0; b < B; ++b) {
        for (int n = 0; n < N; ++n) {
            int offset = (b * N + n) * H;
            float sum_sq = 0.0f;
            for (int h = 0; h < H; ++h) {
                float val = X[offset + h];
                sum_sq += val * val;
            }
            float norm = std::sqrt(sum_sq + kEps);
            for (int h = 0; h < H; ++h) {
                X_hat[offset + h] = X[offset + h] / norm;
            }
        }
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    constexpr int kWarmup = 1;
    constexpr int kRepeat = 10;

    std::vector<std::tuple<int, int, int, int, int>> test_cases = {
        {1, 128, 64, 128, 64},
        {1, 256, 128, 256, 128},
        {2, 512, 128, 512, 128},
        {4, 1024, 256, 1024, 256},
        {8, 2048, 256, 2048, 256},
    };

    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/fused_l2_norm_qk_v2_results.csv");
    ofs << "B,N_q,H_q,N_k,H_k,cpu_ms,gpu_ms,speedup,max_abs_diff_q,max_abs_diff_k,check\n";

    std::cout << "=== Fused L2 Norm Q/K V2 (2 rows/block + ILP + FMA) ===\n";
    std::cout << std::left << std::setw(6) << "B"
              << std::setw(8) << "N_q" << std::setw(8) << "H_q"
              << std::setw(8) << "N_k" << std::setw(8) << "H_k"
              << std::setw(14) << "GPU ms" << std::setw(10) << "Speedup"
              << std::setw(8) << "Check" << "\n";
    std::cout << std::string(64, '-') << "\n";

    for (const auto& tc : test_cases) {
        int B = std::get<0>(tc);
        int N_q = std::get<1>(tc);
        int H_q = std::get<2>(tc);
        int N_k = std::get<3>(tc);
        int H_k = std::get<4>(tc);

        std::vector<float> h_Q(B * N_q * H_q);
        std::vector<float> h_K(B * N_k * H_k);
        std::vector<float> h_Q_hat_cpu(B * N_q * H_q);
        std::vector<float> h_K_hat_cpu(B * N_k * H_k);
        std::vector<float> h_Q_hat_gpu(B * N_q * H_q);
        std::vector<float> h_K_hat_gpu(B * N_k * H_k);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        auto rand_fill = [&](std::vector<float>& v) {
            for (auto& x : v) x = dist(gen);
        };
        rand_fill(h_Q);
        rand_fill(h_K);

        auto t0 = std::chrono::high_resolution_clock::now();
        L2Norm_CPU(h_Q.data(), h_Q_hat_cpu.data(), B, N_q, H_q);
        L2Norm_CPU(h_K.data(), h_K_hat_cpu.data(), B, N_k, H_k);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        float *d_Q, *d_K, *d_Q_hat, *d_K_hat;
        CHECK_CUDA(cudaMalloc(&d_Q, h_Q.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_K, h_K.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_Q_hat, h_Q_hat_gpu.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_K_hat, h_K_hat_gpu.size() * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), h_Q.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), h_K.size() * sizeof(float), cudaMemcpyHostToDevice));

        for (int w = 0; w < kWarmup; ++w) {
            RunGpuV2(d_Q, d_K, d_Q_hat, d_K_hat, B, N_q, H_q, N_k, H_k);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        std::vector<float> gpu_times;
        gpu_times.reserve(kRepeat);

        for (int rep = 0; rep < kRepeat; ++rep) {
            CHECK_CUDA(cudaEventRecord(s));
            RunGpuV2(d_Q, d_K, d_Q_hat, d_K_hat, B, N_q, H_q, N_k, H_k);
            CHECK_CUDA(cudaEventRecord(e));
            CHECK_CUDA(cudaEventSynchronize(e));
            CHECK_CUDA(cudaGetLastError());
            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
            gpu_times.push_back(ms);
        }

        std::sort(gpu_times.begin(), gpu_times.end());
        float gpu_ms = 0.0f;
        if (gpu_times.size() > 2) {
            for (size_t t = 1; t + 1 < gpu_times.size(); ++t) gpu_ms += gpu_times[t];
            gpu_ms /= static_cast<float>(gpu_times.size() - 2);
        } else {
            for (float t : gpu_times) gpu_ms += t;
            gpu_ms /= static_cast<float>(gpu_times.size());
        }

        CHECK_CUDA(cudaMemcpy(h_Q_hat_gpu.data(), d_Q_hat,
                              h_Q_hat_gpu.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_K_hat_gpu.data(), d_K_hat,
                              h_K_hat_gpu.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));

        double max_diff_q = common::MaxAbsDiff(h_Q_hat_cpu, h_Q_hat_gpu);
        double max_diff_k = common::MaxAbsDiff(h_K_hat_cpu, h_K_hat_gpu);
        bool ok = (max_diff_q < 1e-4f && max_diff_k < 1e-4f);
        const char* check = ok ? "PASS" : "FAIL";

        CHECK_CUDA(cudaEventDestroy(s));
        CHECK_CUDA(cudaEventDestroy(e));
        CHECK_CUDA(cudaFree(d_Q));
        CHECK_CUDA(cudaFree(d_K));
        CHECK_CUDA(cudaFree(d_Q_hat));
        CHECK_CUDA(cudaFree(d_K_hat));

        double speedup = (gpu_ms > 0) ? cpu_ms / gpu_ms : 0;
        std::cout << std::left << std::setw(6) << B
                  << std::setw(8) << N_q << std::setw(8) << H_q
                  << std::setw(8) << N_k << std::setw(8) << H_k
                  << std::fixed << std::setprecision(4) << std::setw(14) << gpu_ms
                  << std::setw(10) << std::setprecision(2) << speedup
                  << std::setw(8) << check << "\n";

        ofs << B << "," << N_q << "," << H_q << "," << N_k << "," << H_k << ","
            << cpu_ms << "," << gpu_ms << "," << speedup << ","
            << max_diff_q << "," << max_diff_k << "," << check << "\n";
    }

    std::cout << "\nResults saved to data/results/fused_l2_norm_qk_v2_results.csv\n";
    return 0;
}
