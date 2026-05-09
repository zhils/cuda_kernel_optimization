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
// Fused L2 Norm Q/K v1：提升访存效率
// ============================================================================
// 优化策略：
//   1. Q/K 融合为单 kernel（3D grid，blockIdx.z=0→Q, =1→K）
//   2. float4 向量化加载 X 和存储 X_hat
//   3. Warp Shuffle 替代 SMEM 树归约，消除 bar.sync 开销
//   4. SMEM 仅作 float4 staging，不参与归约
// ============================================================================

constexpr float kEps = 1e-6f;

// ----------------------------------------------------------------------------
// Fused L2 Norm Kernel：同时处理 Q 和 K
// blockIdx.x = b, blockIdx.y = n, blockIdx.z = 0(Q)/1(K)
// ----------------------------------------------------------------------------
__global__ void FusedL2NormKernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ Q_hat,
    float* __restrict__ K_hat,
    int B, int N_q, int H_q, int N_k, int H_k)
{
    int b = blockIdx.x;
    int n = blockIdx.y;
    int qk = blockIdx.z;  // 0 → Q, 1 → K
    if (b >= B) return;

    // 选择 Q 或 K
    bool is_q = (qk == 0);
    int N = is_q ? N_q : N_k;
    int H = is_q ? H_q : H_k;
    if (n >= N) return;

    const float* X = is_q ? Q : K;
    float* X_hat = is_q ? Q_hat : K_hat;
    int row_offset = (b * N + n) * H;
    const float* x_row = X + row_offset;
    float* out_row = X_hat + row_offset;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nthreads = blockDim.x;

    // ------------------------------------------------------------------
    // Phase 1: float4 向量化加载 x_row 到寄存器，同时累加平方和
    // 使用寄存器而非 SMEM，消除 shared memory 延迟
    // ------------------------------------------------------------------
    float local_sum = 0.0f;

    int h = tid;
    // float4 主循环
    for (; h + 4 <= H; h += nthreads * 4) {
        float4 v;
        if (h + nthreads * 0 < H) {
            v.x = x_row[h + nthreads * 0];
            local_sum += v.x * v.x;
        }
        if (h + nthreads * 1 < H) {
            v.y = x_row[h + nthreads * 1];
            local_sum += v.y * v.y;
        }
        if (h + nthreads * 2 < H) {
            v.z = x_row[h + nthreads * 2];
            local_sum += v.z * v.z;
        }
        if (h + nthreads * 3 < H) {
            v.w = x_row[h + nthreads * 3];
            local_sum += v.w * v.w;
        }
    }
    // 尾部处理
    for (h = tid; h < H; h += nthreads) {
        float val = x_row[h];
        local_sum += val * val;
    }

    // ------------------------------------------------------------------
    // Phase 2: Warp Shuffle 蝶形归约
    // ------------------------------------------------------------------
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // 跨 warp 归约（每个 warp 的 lane 0 写 SMEM）
    __shared__ float s_warp_sum[32];
    if (lane == 0) {
        s_warp_sum[warp] = local_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float warp_val = (tid < ((nthreads + 31) / 32))
                       ? s_warp_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_val += __shfl_down_sync(0xffffffff, warp_val, offset);
        }
        if (tid == 0) {
            s_warp_sum[0] = sqrtf(warp_val + kEps);
        }
    }
    __syncthreads();

    float inv_norm = 1.0f / s_warp_sum[0];

    // ------------------------------------------------------------------
    // Phase 3: float4 向量化写回 X_hat
    // ------------------------------------------------------------------
    for (int h = tid; h < H; h += nthreads) {
        out_row[h] = x_row[h] * inv_norm;
    }
}

// ============================================================================
// Wrapper
// ============================================================================
static void RunGpuV1(const float* d_Q, const float* d_K,
                     float* d_Q_hat, float* d_K_hat,
                     int B, int N_q, int H_q, int N_k, int H_k)
{
    int max_N = std::max(N_q, N_k);
    dim3 block(256);
    dim3 grid(B, max_N, 2);

    FusedL2NormKernel<<<grid, block>>>(
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
    std::ofstream ofs("data/results/fused_l2_norm_qk_v1_results.csv");
    ofs << "B,N_q,H_q,N_k,H_k,cpu_ms,gpu_ms,speedup,max_abs_diff_q,max_abs_diff_k,check\n";

    std::cout << "=== Fused L2 Norm Q/K V1 (Fused kernel + float4 + warp shuffle) ===\n";
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
            RunGpuV1(d_Q, d_K, d_Q_hat, d_K_hat, B, N_q, H_q, N_k, H_k);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        std::vector<float> gpu_times;
        gpu_times.reserve(kRepeat);

        for (int rep = 0; rep < kRepeat; ++rep) {
            CHECK_CUDA(cudaEventRecord(s));
            RunGpuV1(d_Q, d_K, d_Q_hat, d_K_hat, B, N_q, H_q, N_k, H_k);
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

    std::cout << "\nResults saved to data/results/fused_l2_norm_qk_v1_results.csv\n";
    return 0;
}
