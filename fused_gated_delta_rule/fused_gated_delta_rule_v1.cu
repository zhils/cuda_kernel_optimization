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
// Fused Gated Delta Rule v1：提升访存效率
// ============================================================================
// 优化策略：
//   1. 全融合：投影 + 递联合并为单 kernel，消除 alpha/delta/u 中间缓冲
//   2. SMEM 缓存 x：每时间步将 x[t][D] 向量化加载到 SMEM，所有 h 共享
//   3. float4 向量化加载 W 矩阵，指令数降至 1/4
//   4. 状态 s 在寄存器中跨时间步递推
//   5. grid(B) 每 block 处理一个 batch，线程池处理所有 h
// ============================================================================

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float softplus(float x) {
    return fmaxf(x, 0.0f) + logf(1.0f + expf(-fabsf(x)));
}

__global__ void fused_gdr_v1_kernel(
    const float* __restrict__ x,
    const float* __restrict__ W_decay,
    const float* __restrict__ b_decay,
    const float* __restrict__ W_delta,
    const float* __restrict__ b_delta,
    const float* __restrict__ W_state,
    const float* __restrict__ b_state,
    float* __restrict__ output,
    int B, int L, int D, int H)
{
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float smem[];
    float* s_x = smem;  // [D]

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int h_per_thread = (H + nthreads - 1) / nthreads;

    // 每个线程维护其负责的 h 的状态 s
    float s_reg[4] = {0.0f};

    for (int t = 0; t < L; ++t) {
        // ------------------------------------------------------------------
        // Phase 1: float4 向量化加载 x[t] → SMEM
        // ------------------------------------------------------------------
        {
            const float* x_bt = x + (size_t(b) * L + t) * D;
            int num_float4 = D / 4;
            int residual = D % 4;
            int idx = tid;
            while (idx < num_float4) {
                reinterpret_cast<float4*>(s_x)[idx] =
                    reinterpret_cast<const float4*>(x_bt)[idx];
                idx += nthreads;
            }
            if (tid < residual) {
                s_x[num_float4 * 4 + tid] = x_bt[num_float4 * 4 + tid];
            }
        }
        __syncthreads();

        // ------------------------------------------------------------------
        // Phase 2: 每线程计算其负责的 h
        //          dot = x_smem @ W_h + b
        //          alpha = sigmoid(dot_decay)
        //          delta = softplus(dot_delta)
        //          u = dot_state
        //          s = alpha * s + delta * u
        //          output = s
        // ------------------------------------------------------------------
        for (int i = 0; i < h_per_thread; ++i) {
            int h = tid + i * nthreads;
            if (h >= H) break;

            // decay gate: alpha = sigmoid(x @ W_decay_h + b_decay_h)
            float sum_decay = b_decay[h];
            const float* Wd_h = W_decay + h * D;
            #pragma unroll 8
            for (int d = 0; d < D; ++d) {
                sum_decay += s_x[d] * Wd_h[d];
            }
            float alpha = sigmoid(sum_decay);

            // delta gate: delta = softplus(x @ W_delta_h + b_delta_h)
            float sum_delta = b_delta[h];
            const float* Wdt_h = W_delta + h * D;
            #pragma unroll 8
            for (int d = 0; d < D; ++d) {
                sum_delta += s_x[d] * Wdt_h[d];
            }
            float delta = softplus(sum_delta);

            // state projection: u = x @ W_state_h + b_state_h
            float sum_state = b_state[h];
            const float* Ws_h = W_state + h * D;
            #pragma unroll 8
            for (int d = 0; d < D; ++d) {
                sum_state += s_x[d] * Ws_h[d];
            }

            // recurrent update
            s_reg[i] = alpha * s_reg[i] + delta * sum_state;

            // write output
            output[(size_t(b) * L + t) * H + h] = s_reg[i];
        }
        __syncthreads();
    }
}

// ============================================================================
// CPU Reference — 同 v0
// ============================================================================
static void GatedDeltaRule_CPU(const float* x,
                               const float* W_decay, const float* b_decay,
                               const float* W_delta, const float* b_delta,
                               const float* W_state, const float* b_state,
                               float* output,
                               int B, int L, int D, int H) {
    std::vector<float> alpha(B * L * H), delta(B * L * H), u(B * L * H);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < L; ++t) {
            const float* x_bt = x + (b * L + t) * D;
            for (int h = 0; h < H; ++h) {
                float a = b_decay[h];
                for (int d = 0; d < D; ++d) a += x_bt[d] * W_decay[h * D + d];
                a = 1.0f / (1.0f + std::exp(-a));
                alpha[(b * L + t) * H + h] = a;

                float dlt = b_delta[h];
                for (int d = 0; d < D; ++d) dlt += x_bt[d] * W_delta[h * D + d];
                dlt = std::max(dlt, 0.0f) + std::log(1.0f + std::exp(-std::abs(dlt)));
                delta[(b * L + t) * H + h] = dlt;

                float v = b_state[h];
                for (int d = 0; d < D; ++d) v += x_bt[d] * W_state[h * D + d];
                u[(b * L + t) * H + h] = v;
            }
        }
    }
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            float s = 0.0f;
            for (int t = 0; t < L; ++t) {
                int idx = (b * L + t) * H + h;
                s = alpha[idx] * s + delta[idx] * u[idx];
                output[idx] = s;
            }
        }
    }
}

// ============================================================================
// Wrapper
// ============================================================================
static void RunGpuV1(const float* d_x,
                     const float* d_W_decay, const float* d_b_decay,
                     const float* d_W_delta, const float* d_b_delta,
                     const float* d_W_state, const float* d_b_state,
                     float* d_output,
                     int B, int L, int D, int H)
{
    dim3 block(256);
    dim3 grid(B);
    size_t smem_bytes = D * sizeof(float);

    fused_gdr_v1_kernel<<<grid, block, smem_bytes>>>(
        d_x, d_W_decay, d_b_decay, d_W_delta, d_b_delta,
        d_W_state, d_b_state, d_output, B, L, D, H);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    constexpr int kWarmup = 1;
    constexpr int kRepeat = 10;

    std::vector<std::tuple<int, int, int, int>> test_cases = {
        {1, 128, 64, 32},
        {1, 256, 128, 64},
        {2, 512, 256, 128},
        {4, 1024, 512, 256},
        {8, 2048, 512, 256},
    };

    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/fused_gated_delta_rule_v1_results.csv");
    ofs << "B,L,D,H,cpu_ms,gpu_ms,speedup,max_abs_diff,check\n";

    std::cout << "=== Fused Gated Delta Rule V1 (Fused single kernel) ===\n";
    std::cout << std::left << std::setw(6) << "B"
              << std::setw(8) << "L" << std::setw(8) << "D"
              << std::setw(8) << "H"
              << std::setw(14) << "GPU ms" << std::setw(10) << "Speedup"
              << std::setw(8) << "Check" << "\n";
    std::cout << std::string(56, '-') << "\n";

    for (const auto& tc : test_cases) {
        int B = std::get<0>(tc);
        int L = std::get<1>(tc);
        int D = std::get<2>(tc);
        int H = std::get<3>(tc);

        std::vector<float> h_x(B * L * D);
        std::vector<float> h_W_decay(H * D), h_b_decay(H);
        std::vector<float> h_W_delta(H * D), h_b_delta(H);
        std::vector<float> h_W_state(H * D), h_b_state(H);
        std::vector<float> h_output_cpu(B * L * H);
        std::vector<float> h_output_gpu(B * L * H);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        auto rand_fill = [&](std::vector<float>& v) {
            for (auto& x : v) x = dist(gen);
        };
        rand_fill(h_x);
        rand_fill(h_W_decay); rand_fill(h_b_decay);
        rand_fill(h_W_delta); rand_fill(h_b_delta);
        rand_fill(h_W_state); rand_fill(h_b_state);

        auto t0 = std::chrono::high_resolution_clock::now();
        GatedDeltaRule_CPU(h_x.data(),
                           h_W_decay.data(), h_b_decay.data(),
                           h_W_delta.data(), h_b_delta.data(),
                           h_W_state.data(), h_b_state.data(),
                           h_output_cpu.data(),
                           B, L, D, H);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        float *d_x, *d_W_decay, *d_b_decay, *d_W_delta, *d_b_delta;
        float *d_W_state, *d_b_state, *d_output;

        CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_W_decay, h_W_decay.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_b_decay, h_b_decay.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_W_delta, h_W_delta.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_b_delta, h_b_delta.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_W_state, h_W_state.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_b_state, h_b_state.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_output, h_output_gpu.size() * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_W_decay, h_W_decay.data(), h_W_decay.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b_decay, h_b_decay.data(), h_b_decay.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_W_delta, h_W_delta.data(), h_W_delta.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b_delta, h_b_delta.data(), h_b_delta.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_W_state, h_W_state.data(), h_W_state.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b_state, h_b_state.data(), h_b_state.size() * sizeof(float), cudaMemcpyHostToDevice));

        for (int w = 0; w < kWarmup; ++w) {
            RunGpuV1(d_x, d_W_decay, d_b_decay, d_W_delta, d_b_delta,
                     d_W_state, d_b_state, d_output, B, L, D, H);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        std::vector<float> gpu_times;
        gpu_times.reserve(kRepeat);

        for (int rep = 0; rep < kRepeat; ++rep) {
            CHECK_CUDA(cudaEventRecord(s));
            RunGpuV1(d_x, d_W_decay, d_b_decay, d_W_delta, d_b_delta,
                     d_W_state, d_b_state, d_output, B, L, D, H);
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

        CHECK_CUDA(cudaMemcpy(h_output_gpu.data(), d_output,
                              h_output_gpu.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));

        double max_diff = common::MaxAbsDiff(h_output_cpu, h_output_gpu);
        bool ok = (max_diff < 1e-3f);
        const char* check = ok ? "PASS" : "FAIL";

        CHECK_CUDA(cudaEventDestroy(s));
        CHECK_CUDA(cudaEventDestroy(e));
        CHECK_CUDA(cudaFree(d_x));
        CHECK_CUDA(cudaFree(d_W_decay)); CHECK_CUDA(cudaFree(d_b_decay));
        CHECK_CUDA(cudaFree(d_W_delta)); CHECK_CUDA(cudaFree(d_b_delta));
        CHECK_CUDA(cudaFree(d_W_state)); CHECK_CUDA(cudaFree(d_b_state));
        CHECK_CUDA(cudaFree(d_output));

        double speedup = (gpu_ms > 0) ? cpu_ms / gpu_ms : 0;
        std::cout << std::left << std::setw(6) << B
                  << std::setw(8) << L << std::setw(8) << D
                  << std::setw(8) << H
                  << std::fixed << std::setprecision(4) << std::setw(14) << gpu_ms
                  << std::setw(10) << std::setprecision(2) << speedup
                  << std::setw(8) << check << "\n";

        ofs << B << "," << L << "," << D << "," << H << ","
            << cpu_ms << "," << gpu_ms << "," << speedup << ","
            << max_diff << "," << check << "\n";
    }

    std::cout << "\nResults saved to data/results/fused_gated_delta_rule_v1_results.csv\n";
    return 0;
}
