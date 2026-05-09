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
// Fused Gated Delta Rule v2：提升计算强度
// ============================================================================
// 优化策略（相对 v1）：
//   1. 双 head ILP：每线程处理 2 个 h，独立累加器交错调度
//   2. __fmaf_rn 替代 a*b+c：融合乘加，指令数减半
//   3. float4 向量化加载 W 矩阵：一次加载 4 个 weight
//   4. 寄存器缓存 x_smem 值：消除重复 SMEM 读取
//   5. #pragma unroll 展开内层 dot product 循环
// ============================================================================

__device__ float sigmoid_fma(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ float softplus_fma(float x) {
    return fmaxf(x, 0.0f) + __logf(1.0f + __expf(-fabsf(x)));
}

// 单线程计算 2 个 h 的 3 路投影 + 递推（ILP 版本）
// 使用 float4 加载 W + __fmaf_rn 累加
__device__ void compute_two_heads_fma(
    const float* __restrict__ s_x,
    const float* __restrict__ Wd0,
    const float* __restrict__ Wdt0,
    const float* __restrict__ Ws0,
    const float* __restrict__ Wd1,
    const float* __restrict__ Wdt1,
    const float* __restrict__ Ws1,
    float bd0, float bdt0, float bs0,
    float bd1, float bdt1, float bs1,
    float& s0, float& s1,
    int D)
{
    // 双路 ILP：两个 h 的累加器交错
    float sd0 = bd0, sd1 = bd1;
    float sdt0 = bdt0, sdt1 = bdt1;
    float ss0 = bs0, ss1 = bs1;

    int d = 0;
    #pragma unroll 4
    for (; d + 4 <= D; d += 4) {
        float4 xv = reinterpret_cast<const float4*>(s_x)[d / 4];

        float4 wd0 = reinterpret_cast<const float4*>(Wd0)[d / 4];
        sd0 = __fmaf_rn(xv.x, wd0.x, sd0);
        sd0 = __fmaf_rn(xv.y, wd0.y, sd0);
        sd0 = __fmaf_rn(xv.z, wd0.z, sd0);
        sd0 = __fmaf_rn(xv.w, wd0.w, sd0);

        float4 wd1 = reinterpret_cast<const float4*>(Wd1)[d / 4];
        sd1 = __fmaf_rn(xv.x, wd1.x, sd1);
        sd1 = __fmaf_rn(xv.y, wd1.y, sd1);
        sd1 = __fmaf_rn(xv.z, wd1.z, sd1);
        sd1 = __fmaf_rn(xv.w, wd1.w, sd1);

        float4 wdt0 = reinterpret_cast<const float4*>(Wdt0)[d / 4];
        sdt0 = __fmaf_rn(xv.x, wdt0.x, sdt0);
        sdt0 = __fmaf_rn(xv.y, wdt0.y, sdt0);
        sdt0 = __fmaf_rn(xv.z, wdt0.z, sdt0);
        sdt0 = __fmaf_rn(xv.w, wdt0.w, sdt0);

        float4 wdt1 = reinterpret_cast<const float4*>(Wdt1)[d / 4];
        sdt1 = __fmaf_rn(xv.x, wdt1.x, sdt1);
        sdt1 = __fmaf_rn(xv.y, wdt1.y, sdt1);
        sdt1 = __fmaf_rn(xv.z, wdt1.z, sdt1);
        sdt1 = __fmaf_rn(xv.w, wdt1.w, sdt1);

        float4 ws0 = reinterpret_cast<const float4*>(Ws0)[d / 4];
        ss0 = __fmaf_rn(xv.x, ws0.x, ss0);
        ss0 = __fmaf_rn(xv.y, ws0.y, ss0);
        ss0 = __fmaf_rn(xv.z, ws0.z, ss0);
        ss0 = __fmaf_rn(xv.w, ws0.w, ss0);

        float4 ws1 = reinterpret_cast<const float4*>(Ws1)[d / 4];
        ss1 = __fmaf_rn(xv.x, ws1.x, ss1);
        ss1 = __fmaf_rn(xv.y, ws1.y, ss1);
        ss1 = __fmaf_rn(xv.z, ws1.z, ss1);
        ss1 = __fmaf_rn(xv.w, ws1.w, ss1);
    }
    for (; d < D; ++d) {
        float xv = s_x[d];
        sd0 = __fmaf_rn(xv, Wd0[d], sd0);
        sd1 = __fmaf_rn(xv, Wd1[d], sd1);
        sdt0 = __fmaf_rn(xv, Wdt0[d], sdt0);
        sdt1 = __fmaf_rn(xv, Wdt1[d], sdt1);
        ss0 = __fmaf_rn(xv, Ws0[d], ss0);
        ss1 = __fmaf_rn(xv, Ws1[d], ss1);
    }

    float a0 = sigmoid_fma(sd0), a1 = sigmoid_fma(sd1);
    float d0 = softplus_fma(sdt0), d1 = softplus_fma(sdt1);
    s0 = a0 * s0 + d0 * ss0;
    s1 = a1 * s1 + d1 * ss1;
}

__global__ void fused_gdr_v2_kernel(
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
    float* s_x = smem;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int h_pairs = (H + 2 * nthreads - 1) / (2 * nthreads);

    float s_reg0[2] = {0.0f, 0.0f};
    float s_reg1[2] = {0.0f, 0.0f};

    for (int t = 0; t < L; ++t) {
        // float4 加载 x[t] → SMEM
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

        for (int p = 0; p < h_pairs; ++p) {
            int h0 = (tid * 2 + 0) + p * (2 * nthreads);
            int h1 = (tid * 2 + 1) + p * (2 * nthreads);
            if (h0 >= H) break;

            int idx0 = h0 / 2;
            float s0_val = s_reg0[idx0];
            float s1_val = (h1 < H) ? s_reg1[idx0] : 0.0f;

            compute_two_heads_fma(
                s_x,
                W_decay + h0 * D, W_delta + h0 * D, W_state + h0 * D,
                W_decay + ((h1 < H) ? h1 : h0) * D,
                W_delta + ((h1 < H) ? h1 : h0) * D,
                W_state + ((h1 < H) ? h1 : h0) * D,
                b_decay[h0], b_delta[h0], b_state[h0],
                b_decay[h1 < H ? h1 : h0],
                b_delta[h1 < H ? h1 : h0],
                b_state[h1 < H ? h1 : h0],
                s0_val, s1_val, D
            );

            s_reg0[idx0] = s0_val;
            if (h1 < H) s_reg1[idx0] = s1_val;

            output[(size_t(b) * L + t) * H + h0] = s0_val;
            if (h1 < H) {
                output[(size_t(b) * L + t) * H + h1] = s1_val;
            }
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
static void RunGpuV2(const float* d_x,
                     const float* d_W_decay, const float* d_b_decay,
                     const float* d_W_delta, const float* d_b_delta,
                     const float* d_W_state, const float* d_b_state,
                     float* d_output,
                     int B, int L, int D, int H)
{
    dim3 block(128);
    dim3 grid(B);
    size_t smem_bytes = D * sizeof(float);

    fused_gdr_v2_kernel<<<grid, block, smem_bytes>>>(
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
    std::ofstream ofs("data/results/fused_gated_delta_rule_v2_results.csv");
    ofs << "B,L,D,H,cpu_ms,gpu_ms,speedup,max_abs_diff,check\n";

    std::cout << "=== Fused Gated Delta Rule V2 (ILP + FMA + 2-head/thread) ===\n";
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
            RunGpuV2(d_x, d_W_decay, d_b_decay, d_W_delta, d_b_delta,
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
            RunGpuV2(d_x, d_W_decay, d_b_decay, d_W_delta, d_b_delta,
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

    std::cout << "\nResults saved to data/results/fused_gated_delta_rule_v2_results.csv\n";
    return 0;
}
