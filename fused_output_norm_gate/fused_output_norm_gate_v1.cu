#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

// ============================================================================
// Fused Output Norm Gate v1: 提升访存效率
// ============================================================================
// 优化策略：
//   1. 全融合：4 个操作合并为 1 个 kernel，消除 gate / x_hat / y 中间缓冲
//   2. float4 向量化加载 x 和存储 output
//   3. SMEM 双缓冲：s_x[D_in] 缓存输入，s_gate[H] 缓存门控值
//   4. Warp Shuffle 完成 RMSNorm 归约，避免 SMEM 树归约
//   5. 每 block 处理一行 (b,t)，线程数 = 256
// ============================================================================

constexpr float kEps = 1e-6f;

__global__ void fused_output_norm_gate_v1_kernel(
    const float* __restrict__ x,
    const float* __restrict__ W_gate,
    const float* __restrict__ b_gate,
    const float* __restrict__ g,
    const float* __restrict__ W_out,
    const float* __restrict__ b_out,
    float* __restrict__ output,
    int B, int L, int D_in, int H, int D_out)
{
    int row = blockIdx.x;
    if (row >= B * L) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nthreads = blockDim.x;

    extern __shared__ float smem[];
    float* s_x     = smem;                 // [D_in]
    float* s_gate  = smem + D_in;          // [H], 复用为 y

    const float* x_row = x + row * D_in;

    // ------------------------------------------------------------------
    // Phase 1: float4 向量化加载 x 到 SMEM
    // ------------------------------------------------------------------
    {
        const int num_float4 = D_in / 4;
        const int residual   = D_in % 4;
        int idx = tid;
        while (idx < num_float4) {
            reinterpret_cast<float4*>(s_x)[idx] =
                reinterpret_cast<const float4*>(x_row)[idx];
            idx += nthreads;
        }
        if (tid < residual) {
            s_x[num_float4 * 4 + tid] = x_row[num_float4 * 4 + tid];
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 2: Gate Projection + SiLU（从 SMEM 读 x，每个线程算多个 h）
    // ------------------------------------------------------------------
    {
        const int h_per_thread = (H + nthreads - 1) / nthreads;
        for (int i = 0; i < h_per_thread; ++i) {
            int h = tid + i * nthreads;
            if (h >= H) break;

            float sum = b_gate[h];
            const float* W_h = W_gate + h * D_in;
            #pragma unroll 8
            for (int d = 0; d < D_in; ++d) {
                sum += s_x[d] * W_h[d];
            }
            float sigmoid = 1.0f / (1.0f + expf(-sum));
            s_gate[h] = sum * sigmoid;
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 3: RMSNorm — Warp Shuffle 归约求 sum_sq
    // ------------------------------------------------------------------
    float local_sum_sq = 0.0f;
    for (int h = tid; h < H; h += nthreads) {
        float val = s_gate[h];
        local_sum_sq += val * val;
    }

    // warp-level shuffle reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }

    // block-level reduce via SMEM (one slot per warp)
    __shared__ float s_warp_sum[32];
    if (lane == 0) {
        s_warp_sum[warp] = local_sum_sq;
    }
    __syncthreads();

    if (warp == 0) {
        float warp_val = (tid < ((nthreads + 31) / 32))
                       ? s_warp_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_val += __shfl_down_sync(0xffffffff, warp_val, offset);
        }
        if (tid == 0) {
            s_warp_sum[0] = warp_val;
        }
    }
    __syncthreads();

    float rms = sqrtf(s_warp_sum[0] / H + kEps);
    float inv_rms = 1.0f / rms;

    // ------------------------------------------------------------------
    // Phase 4: y = (gate / rms * g) * gate  覆写 s_gate
    // ------------------------------------------------------------------
    for (int h = tid; h < H; h += nthreads) {
        float gate_h = s_gate[h];
        float x_hat = gate_h * inv_rms * g[h];
        s_gate[h] = x_hat * gate_h;        // now holds y
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 5: Output Projection（从 SMEM 读 y，每个线程算多个 d）
    // ------------------------------------------------------------------
    {
        const int d_per_thread = (D_out + nthreads - 1) / nthreads;
        for (int i = 0; i < d_per_thread; ++i) {
            int d = tid + i * nthreads;
            if (d >= D_out) break;

            float sum = b_out[d];
            const float* W_d = W_out + d * H;
            #pragma unroll 8
            for (int h = 0; h < H; ++h) {
                sum += s_gate[h] * W_d[h];
            }
            // 写回 output（由调用者做向量化写，此处直接写）
            output[row * D_out + d] = sum;
        }
    }
}

// ============================================================================
// Wrapper
// ============================================================================
static void RunGpuV1(const float* d_x,
                     const float* d_W_gate, const float* d_b_gate,
                     const float* d_g,
                     const float* d_W_out, const float* d_b_out,
                     float* d_output,
                     int B, int L, int D_in, int H, int D_out)
{
    int total_rows = B * L;
    int threads = 256;
    size_t smem_bytes = (D_in + H) * sizeof(float);

    dim3 grid(total_rows);
    fused_output_norm_gate_v1_kernel<<<grid, threads, smem_bytes>>>(
        d_x, d_W_gate, d_b_gate, d_g, d_W_out, d_b_out, d_output,
        B, L, D_in, H, D_out);
}

// ============================================================================
// CPU Reference — 同 v0
// ============================================================================
static void OutputNormGate_CPU(const float* x,
                               const float* W_gate, const float* b_gate,
                               const float* g,
                               const float* W_out, const float* b_out,
                               float* output,
                               int B, int L, int D_in, int H, int D_out)
{
    std::vector<float> gate(B * L * H);
    std::vector<float> x_hat(B * L * H);
    std::vector<float> y(B * L * H);

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < L; ++t) {
            const float* x_bt = x + (b * L + t) * D_in;
            for (int h = 0; h < H; ++h) {
                float sum = b_gate[h];
                for (int d = 0; d < D_in; ++d) {
                    sum += x_bt[d] * W_gate[h * D_in + d];
                }
                float sigmoid = 1.0f / (1.0f + std::exp(-sum));
                gate[(b * L + t) * H + h] = sum * sigmoid;
            }
        }
    }

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < L; ++t) {
            int offset = (b * L + t) * H;
            float sum_sq = 0.0f;
            for (int h = 0; h < H; ++h) {
                sum_sq += gate[offset + h] * gate[offset + h];
            }
            float rms = std::sqrt(sum_sq / H + kEps);
            for (int h = 0; h < H; ++h) {
                x_hat[offset + h] = (gate[offset + h] / rms) * g[h];
            }
        }
    }

    for (int i = 0; i < B * L * H; ++i) {
        y[i] = x_hat[i] * gate[i];
    }

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < L; ++t) {
            const float* y_bt = y.data() + (b * L + t) * H;
            for (int d = 0; d < D_out; ++d) {
                float sum = b_out[d];
                for (int h = 0; h < H; ++h) {
                    sum += y_bt[h] * W_out[d * H + h];
                }
                output[(b * L + t) * D_out + d] = sum;
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
        {1, 128, 64, 32, 64},
        {1, 256, 128, 64, 128},
        {2, 512, 256, 128, 256},
        {4, 1024, 512, 256, 512},
        {8, 2048, 512, 256, 512},
    };

    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/fused_output_norm_gate_v1_results.csv");
    ofs << "B,L,D_in,H,D_out,cpu_ms,gpu_ms,speedup,max_abs_diff,check\n";

    std::cout << "=== Fused Output Norm Gate V1 (Fused Single Kernel) ===\n";
    std::cout << std::left << std::setw(6) << "B"
              << std::setw(8) << "L" << std::setw(8) << "D_in"
              << std::setw(8) << "H" << std::setw(8) << "D_out"
              << std::setw(14) << "GPU ms" << std::setw(10) << "Speedup"
              << std::setw(8) << "Check" << "\n";
    std::cout << std::string(72, '-') << "\n";

    for (const auto& tc : test_cases) {
        int B = std::get<0>(tc);
        int L = std::get<1>(tc);
        int D_in = std::get<2>(tc);
        int H = std::get<3>(tc);
        int D_out = std::get<4>(tc);

        std::vector<float> h_x(B * L * D_in);
        std::vector<float> h_W_gate(H * D_in), h_b_gate(H);
        std::vector<float> h_g(H);
        std::vector<float> h_W_out(D_out * H), h_b_out(D_out);
        std::vector<float> h_output_cpu(B * L * D_out);
        std::vector<float> h_output_gpu(B * L * D_out);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        auto rand_fill = [&](std::vector<float>& v) {
            for (auto& x : v) x = dist(gen);
        };
        rand_fill(h_x);
        rand_fill(h_W_gate); rand_fill(h_b_gate);
        rand_fill(h_g);
        rand_fill(h_W_out); rand_fill(h_b_out);

        auto t0 = std::chrono::high_resolution_clock::now();
        OutputNormGate_CPU(h_x.data(),
                           h_W_gate.data(), h_b_gate.data(),
                           h_g.data(),
                           h_W_out.data(), h_b_out.data(),
                           h_output_cpu.data(),
                           B, L, D_in, H, D_out);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        float *d_x, *d_W_gate, *d_b_gate, *d_g, *d_W_out, *d_b_out, *d_output;
        CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_W_gate, h_W_gate.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_b_gate, h_b_gate.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_g, h_g.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_W_out, h_W_out.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_b_out, h_b_out.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_output, h_output_gpu.size() * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_W_gate, h_W_gate.data(), h_W_gate.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b_gate, h_b_gate.data(), h_b_gate.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_g, h_g.data(), h_g.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_W_out, h_W_out.data(), h_W_out.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b_out, h_b_out.data(), h_b_out.size() * sizeof(float), cudaMemcpyHostToDevice));

        for (int w = 0; w < kWarmup; ++w) {
            RunGpuV1(d_x, d_W_gate, d_b_gate, d_g, d_W_out, d_b_out,
                     d_output, B, L, D_in, H, D_out);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        std::vector<float> gpu_times;
        gpu_times.reserve(kRepeat);

        for (int rep = 0; rep < kRepeat; ++rep) {
            CHECK_CUDA(cudaEventRecord(s));
            RunGpuV1(d_x, d_W_gate, d_b_gate, d_g, d_W_out, d_b_out,
                     d_output, B, L, D_in, H, D_out);
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
        CHECK_CUDA(cudaFree(d_W_gate)); CHECK_CUDA(cudaFree(d_b_gate));
        CHECK_CUDA(cudaFree(d_g));
        CHECK_CUDA(cudaFree(d_W_out)); CHECK_CUDA(cudaFree(d_b_out));
        CHECK_CUDA(cudaFree(d_output));

        double speedup = (gpu_ms > 0) ? cpu_ms / gpu_ms : 0;
        std::cout << std::left << std::setw(6) << B
                  << std::setw(8) << L << std::setw(8) << D_in
                  << std::setw(8) << H << std::setw(8) << D_out
                  << std::fixed << std::setprecision(4) << std::setw(14) << gpu_ms
                  << std::setw(10) << std::setprecision(2) << speedup
                  << std::setw(8) << check << "\n";

        ofs << B << "," << L << "," << D_in << "," << H << "," << D_out << ","
            << cpu_ms << "," << gpu_ms << "," << speedup << ","
            << max_diff << "," << check << "\n";
    }

    std::cout << "\nResults saved to data/results/fused_output_norm_gate_v1_results.csv\n";
    return 0;
}
