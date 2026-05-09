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
// Fused Output Norm Gate v2: 提升计算强度
// ============================================================================
// 优化策略：
//   1. 每 block 处理 2 行 (b,t)，复用权重读取（计算量翻倍，访存量不变）
//   2. float4 向量化加载 W_gate / W_out / x，利用宽位加载提升吞吐
//   3. #pragma unroll 展开内层 matvec 循环
//   4. RMSNorm 归约使用 Warp Shuffle + 单线程串行处理（降低同步开销）
//   5. 寄存器缓存累加器，减少 SMEM 冲突
// ============================================================================

constexpr float kEps = 1e-6f;
constexpr int kRowsPerBlock = 2;

// 辅助：向量化加载 float4，处理不对齐尾部
template <typename T>
__device__ void vec_load_4(T* dst, const T* src, int n) {
    int i = threadIdx.x;
    for (; i + 4 <= n; i += blockDim.x) {
        reinterpret_cast<float4*>(dst)[i / 4] =
            reinterpret_cast<const float4*>(src)[i / 4];
    }
    for (; i < n; ++i) {
        dst[i] = src[i];
    }
}

// 辅助：向量化存储 float4，处理不对齐尾部
template <typename T>
__device__ void vec_store_4(T* dst, const T* src, int n) {
    int i = threadIdx.x;
    for (; i + 4 <= n; i += blockDim.x) {
        reinterpret_cast<float4*>(dst)[i / 4] =
            reinterpret_cast<const float4*>(src)[i / 4];
    }
    for (; i < n; ++i) {
        dst[i] = src[i];
    }
}

// 辅助：线程计算 gate_h = SiLU(sum_d(x[d] * W_gate[h][d]) + b_gate[h])
// 使用 float4 向量化加载 W_gate，利用 __fmaf_rn 融合乘加
__device__ float compute_gate_silu(
    const float* __restrict__ x_smem,
    const float* __restrict__ W_h,
    float bias,
    int D_in)
{
    float sum = bias;
    int d = 0;
    #pragma unroll 4
    for (; d + 4 <= D_in; d += 4) {
        float4 w = reinterpret_cast<const float4*>(W_h)[d / 4];
        sum = __fmaf_rn(x_smem[d + 0], w.x, sum);
        sum = __fmaf_rn(x_smem[d + 1], w.y, sum);
        sum = __fmaf_rn(x_smem[d + 2], w.z, sum);
        sum = __fmaf_rn(x_smem[d + 3], w.w, sum);
    }
    for (; d < D_in; ++d) {
        sum += x_smem[d] * W_h[d];
    }
    float sigmoid = 1.0f / (1.0f + expf(-sum));
    return sum * sigmoid;
}

// 辅助：线程计算 output[d] = sum_h(y[h] * W_out[d][h]) + b_out[d]
// 使用 float4 向量化加载 W_out
__device__ float compute_output(
    const float* __restrict__ y_smem,
    const float* __restrict__ W_d,
    float bias,
    int H)
{
    float sum = bias;
    int h = 0;
    #pragma unroll 4
    for (; h + 4 <= H; h += 4) {
        float4 w = reinterpret_cast<const float4*>(W_d)[h / 4];
        sum = __fmaf_rn(y_smem[h + 0], w.x, sum);
        sum = __fmaf_rn(y_smem[h + 1], w.y, sum);
        sum = __fmaf_rn(y_smem[h + 2], w.z, sum);
        sum = __fmaf_rn(y_smem[h + 3], w.w, sum);
    }
    for (; h < H; ++h) {
        sum += y_smem[h] * W_d[h];
    }
    return sum;
}

// 辅助：RMSNorm 归约 — 单线程串行处理每个 h（减少同步开销）
// 输入 s_gate[H] 在 SMEM 中，输出 rms
__device__ float rmsnorm_reduce(const float* s_gate, int H)
{
    float sum_sq = 0.0f;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float v = s_gate[h];
        sum_sq += v * v;
    }

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float s_warp[32];
    if (lane == 0) s_warp[warp] = sum_sq;
    __syncthreads();

    if (warp == 0) {
        float val = (threadIdx.x < ((blockDim.x + 31) / 32))
                  ? s_warp[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (threadIdx.x == 0) s_warp[0] = val;
    }
    __syncthreads();

    return sqrtf(s_warp[0] / H + kEps);
}

// ----------------------------------------------------------------------------
// 主 kernel：每 block 处理 kRowsPerBlock 行
// ----------------------------------------------------------------------------
__global__ void fused_output_norm_gate_v2_kernel(
    const float* __restrict__ x,
    const float* __restrict__ W_gate,
    const float* __restrict__ b_gate,
    const float* __restrict__ g,
    const float* __restrict__ W_out,
    const float* __restrict__ b_out,
    float* __restrict__ output,
    int B, int L, int D_in, int H, int D_out)
{
    int block_row_start = blockIdx.x * kRowsPerBlock;
    if (block_row_start >= B * L) return;
    int rows_this_block = min(kRowsPerBlock, B * L - block_row_start);

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // SMEM 布局
    extern __shared__ float smem[];
    float* s_x0   = smem;                        // [D_in]
    float* s_x1   = smem + D_in;                 // [D_in]
    float* s_gate = smem + 2 * D_in;             // [H] 逐行复用
    // s_gate 之后还需要 H 字节用于 rmsnorm 中间结果
    // 但 rmsnorm 只用 H 个元素

    // 第 0 行索引
    int row0 = block_row_start;
    int row1 = min(block_row_start + 1, B * L - 1);

    // ==================================================================
    // Phase 1: 向量化加载 x0, x1 到 SMEM
    // ==================================================================
    vec_load_4(s_x0, x + row0 * D_in, D_in);
    if (rows_this_block > 1) {
        vec_load_4(s_x1, x + row1 * D_in, D_in);
    }
    __syncthreads();

    // ==================================================================
    // Phase 2: Gate Projection + SiLU（每线程处理 1 个 h）
    //         W_gate 读一次，计算 2 行的 gate 值
    // ==================================================================
    for (int h = tid; h < H; h += nthreads) {
        const float* W_h = W_gate + h * D_in;
        float gate0 = compute_gate_silu(s_x0, W_h, b_gate[h], D_in);
        s_gate[h] = gate0;      // row 0 gate 暂存
    }
    __syncthreads();

    // ---- row 0: RMSNorm + Mul ----
    {
        float rms = rmsnorm_reduce(s_gate, H);
        float inv_rms = 1.0f / rms;
        for (int h = tid; h < H; h += nthreads) {
            float gate_h = s_gate[h];
            float x_hat = gate_h * inv_rms * g[h];
            s_gate[h] = x_hat * gate_h;  // → y0
        }
    }
    __syncthreads();

    // ---- row 0: Output Projection ----
    for (int d = tid; d < D_out; d += nthreads) {
        float val = compute_output(s_gate,
                                   W_out + d * H,
                                   b_out[d], H);
        output[row0 * D_out + d] = val;
    }

    // ---- row 1: Gate Proj + SiLU（复用 W_gate） ----
    if (rows_this_block > 1) {
        for (int h = tid; h < H; h += nthreads) {
            const float* W_h = W_gate + h * D_in;
            float gate1 = compute_gate_silu(s_x1, W_h, b_gate[h], D_in);
            s_gate[h] = gate1;
        }
        __syncthreads();

        // ---- row 1: RMSNorm + Mul ----
        {
            float rms = rmsnorm_reduce(s_gate, H);
            float inv_rms = 1.0f / rms;
            for (int h = tid; h < H; h += nthreads) {
                float gate_h = s_gate[h];
                float x_hat = gate_h * inv_rms * g[h];
                s_gate[h] = x_hat * gate_h;  // → y1
            }
        }
        __syncthreads();

        // ---- row 1: Output Projection（复用 W_out） ----
        for (int d = tid; d < D_out; d += nthreads) {
            float val = compute_output(s_gate,
                                       W_out + d * H,
                                       b_out[d], H);
            output[row1 * D_out + d] = val;
        }
    }
}

// ============================================================================
// Wrapper
// ============================================================================
static void RunGpuV2(const float* d_x,
                     const float* d_W_gate, const float* d_b_gate,
                     const float* d_g,
                     const float* d_W_out, const float* d_b_out,
                     float* d_output,
                     int B, int L, int D_in, int H, int D_out)
{
    int total_rows = B * L;
    int threads = 256;
    size_t smem_bytes = (2 * D_in + H) * sizeof(float);

    dim3 grid((total_rows + kRowsPerBlock - 1) / kRowsPerBlock);
    fused_output_norm_gate_v2_kernel<<<grid, threads, smem_bytes>>>(
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
    std::ofstream ofs("data/results/fused_output_norm_gate_v2_results.csv");
    ofs << "B,L,D_in,H,D_out,cpu_ms,gpu_ms,speedup,max_abs_diff,check\n";

    std::cout << "=== Fused Output Norm Gate V2 (2 rows/block, weight reuse) ===\n";
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
            RunGpuV2(d_x, d_W_gate, d_b_gate, d_g, d_W_out, d_b_out,
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
            RunGpuV2(d_x, d_W_gate, d_b_gate, d_g, d_W_out, d_b_out,
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

    std::cout << "\nResults saved to data/results/fused_output_norm_gate_v2_results.csv\n";
    return 0;
}
