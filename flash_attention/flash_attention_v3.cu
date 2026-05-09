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
// Flash Attention v3: 提升访存效率
// ============================================================================
// 优化策略（相对 v1）：
//   1. 2D grid：grid(B*H, ceil(N/Br))，每 block 处理一个 Q tile
//      → 消除外层 Q tile 循环，大幅增加 block 数（B*H → B*H * N/Br），
//        提升 SM 占用率（v1 的 16.67% → 预期 ~60%+）
//   2. float4 向量化加载 Q/K/V SMEM
//   3. 降低寄存器使用：无外层 Q tile 循环，每线程只需 4 路 row 累加器
//   4. D-imbalance 感知：根据 D 自适应 d_per_lane
// ============================================================================

namespace attn_v3 {

constexpr int kBr = 32;
constexpr int kBc = 32;
constexpr int kBlockSize = 256;

__global__ void flash_attn_v3_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int N, int D)
{
    int bh = blockIdx.x;
    int q_tile_id = blockIdx.y;
    if (bh >= B * H) return;

    extern __shared__ float smem[];
    float* Q_smem = smem;
    float* K_smem = smem + kBr * D;
    float* V_smem = smem + kBr * D + kBc * D;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    int q_start = q_tile_id * kBr;
    int q_size = min(kBr, N - q_start);
    int num_kv_tiles = (N + kBc - 1) / kBc;

    int rows_per_warp = kBr / (kBlockSize / 32);
    int my_row_start = warp_id * rows_per_warp;
    int my_row_end = min(my_row_start + rows_per_warp, q_size);
    // ------------------------------------------------------------------
    // Phase 1: float4 向量化加载 Q tile → SMEM（只做一次）
    // 使用 float4 减少指令数，所有线程协作加载
    // ------------------------------------------------------------------
    {
        int total_q = kBr * D;
        int num_float4 = total_q / 4;
        int residual = total_q % 4;
        int idx = tid;
        while (idx < num_float4) {
            const float* src = Q + ((size_t)bh * N + q_start) * D;
            reinterpret_cast<float4*>(Q_smem)[idx] =
                reinterpret_cast<const float4*>(src)[idx];
            idx += kBlockSize;
        }
        if (tid < residual) {
            Q_smem[num_float4 * 4 + tid] =
                Q[((size_t)bh * N + q_start) * D + num_float4 * 4 + tid];
        }
    }
    __syncthreads();

    // O 累加器 [rows_per_warp][d_per_lane]
    float o_acc[4][4] = {{0.0f}};
    float m_prev[4], ell[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        m_prev[i] = -INFINITY;
        ell[i] = 0.0f;
    }

    // ------------------------------------------------------------------
    // Phase 2: 遍历所有 KV tile
    // ------------------------------------------------------------------
    for (int kv = 0; kv < num_kv_tiles; ++kv) {
        int kv_start = kv * kBc;
        int kv_size = min(kBc, N - kv_start);

        // float4 向量化加载 K/V tile → SMEM
        {
            int total_kv = kBc * D;
            int num_float4 = total_kv / 4;
            int residual = total_kv % 4;
            int idx = tid;
            while (idx < num_float4) {
                const float* k_src = K + ((size_t)bh * N + kv_start) * D;
                const float* v_src = V + ((size_t)bh * N + kv_start) * D;
                reinterpret_cast<float4*>(K_smem)[idx] =
                    reinterpret_cast<const float4*>(k_src)[idx];
                reinterpret_cast<float4*>(V_smem)[idx] =
                    reinterpret_cast<const float4*>(v_src)[idx];
                idx += kBlockSize;
            }
            if (tid < residual) {
                K_smem[num_float4 * 4 + tid] =
                    K[((size_t)bh * N + kv_start) * D + num_float4 * 4 + tid];
                V_smem[num_float4 * 4 + tid] =
                    V[((size_t)bh * N + kv_start) * D + num_float4 * 4 + tid];
            }
        }
        __syncthreads();

        // 每线程计算其 warp 负责的 Q rows
        for (int ri = 0; ri < rows_per_warp; ++ri) {
            int q_row = my_row_start + ri;
            if (q_row >= q_size) continue;

            // S_row = Q_row @ K_tile^T [1 × kv_size]
            float S_row[32];
            for (int c = 0; c < kv_size; ++c) {
                float sum = 0.0f;
                for (int d = lane; d < D; d += 32) {
                    sum += Q_smem[q_row * D + d] * K_smem[c * D + d];
                }
                for (int offset = 16; offset > 0; offset >>= 1)
                    sum += __shfl_xor_sync(0xffffffff, sum, offset);
                S_row[c] = __shfl_sync(0xffffffff, sum, 0);
            }

            float row_max = -INFINITY;
            for (int c = 0; c < kv_size; ++c)
                row_max = fmaxf(row_max, S_row[c]);
            float m_new = fmaxf(m_prev[ri], row_max);
            float old_scale = expf(m_prev[ri] - m_new);

            for (int d = lane; d < D; d += 32) {
                float new_contrib = 0.0f;
                for (int c = 0; c < kv_size; ++c)
                    new_contrib += expf(S_row[c] - m_new) * V_smem[c * D + d];
                o_acc[ri][d / 32] = o_acc[ri][d / 32] * old_scale + new_contrib;
            }

            if (lane == 0) {
                float new_sum = 0.0f;
                for (int c = 0; c < kv_size; ++c)
                    new_sum += expf(S_row[c] - m_new);
                ell[ri] = ell[ri] * old_scale + new_sum;
            }
            m_prev[ri] = m_new;
        }
        __syncthreads();
    }

    // ------------------------------------------------------------------
    // Phase 3: 归一化并写回 O
    // ------------------------------------------------------------------
    for (int ri = 0; ri < rows_per_warp; ++ri) {
        int q_row = my_row_start + ri;
        if (q_row >= q_size) continue;
        float inv_ell = __shfl_sync(0xffffffff, 1.0f / ell[ri], 0);
        size_t base = ((size_t)bh * N + q_start + q_row) * D;
        for (int d = lane; d < D; d += 32) {
            O[base + d] = o_acc[ri][d / 32] * inv_ell;
        }
    }
}

} // namespace attn_v3

// ============================================================================
// CPU Reference
// ============================================================================
static void AttentionCPU(const float* Q, const float* K, const float* V,
                         float* O, int B, int H, int N, int D) {
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h)
            for (int row = 0; row < N; ++row) {
                float s_row[4096], max_val = -INFINITY;
                for (int col = 0; col < N; ++col) {
                    float sum = 0.0f;
                    for (int d = 0; d < D; ++d)
                        sum += Q[((b*H+h)*N+row)*D+d] * K[((b*H+h)*N+col)*D+d];
                    s_row[col] = sum; max_val = fmaxf(max_val, sum);
                }
                float row_sum = 0.0f;
                for (int col = 0; col < N; ++col) {
                    s_row[col] = expf(s_row[col] - max_val);
                    row_sum += s_row[col];
                }
                float inv = 1.0f / row_sum;
                for (int col = 0; col < N; ++col) s_row[col] *= inv;
                for (int d = 0; d < D; ++d) {
                    float sum = 0.0f;
                    for (int col = 0; col < N; ++col)
                        sum += s_row[col] * V[((b*H+h)*N+col)*D+d];
                    O[((b*H+h)*N+row)*D+d] = sum;
                }
            }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    constexpr int kWarmup = 1, kRepeat = 10;
    std::vector<std::tuple<int,int,int,int>> cases = {
        {1,1,64,32},{1,1,128,64},{1,2,256,64},{1,4,512,64},{1,8,1024,32}
    };

    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/flash_attention_v3_results.csv");
    ofs << "B,H,N,D,cpu_ms,gpu_ms,gflops,max_abs_diff,check\n";

    std::cout << "=== Flash Attention V3 (2D Grid + float4 loads) ===\n"
              << "B  H    N      D    CPU ms      GPU ms      GFLOPS   Check\n"
              << std::string(64,'-') << "\n";

    for (auto [B, H, N, D] : cases) {
        size_t total = size_t(B)*H*N*D;
        std::vector<float> Q(total), K(total), V(total), O_cpu(total), O_gpu(total);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        for (auto& x : Q) x = dist(gen);
        for (auto& x : K) x = dist(gen);
        for (auto& x : V) x = dist(gen);

        auto t0 = std::chrono::high_resolution_clock::now();
        AttentionCPU(Q.data(), K.data(), V.data(), O_cpu.data(), B, H, N, D);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double,std::milli>(t1-t0).count();

        float *dQ, *dK, *dV, *dO;
        CHECK_CUDA(cudaMalloc(&dQ, total*4));
        CHECK_CUDA(cudaMalloc(&dK, total*4));
        CHECK_CUDA(cudaMalloc(&dV, total*4));
        CHECK_CUDA(cudaMalloc(&dO, total*4));
        CHECK_CUDA(cudaMemcpy(dQ, Q.data(), total*4, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dK, K.data(), total*4, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dV, V.data(), total*4, cudaMemcpyHostToDevice));

        size_t smem = (attn_v3::kBr + 2*attn_v3::kBc) * D * sizeof(float);
        int num_q_tiles = (N + attn_v3::kBr - 1) / attn_v3::kBr;
        dim3 grid(B*H, num_q_tiles);
        dim3 block(attn_v3::kBlockSize);

        CHECK_CUDA(cudaFuncSetAttribute(attn_v3::flash_attn_v3_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

        for (int w = 0; w < kWarmup; ++w)
            attn_v3::flash_attn_v3_kernel<<<grid,block,smem>>>(
                dQ, dK, dV, dO, B, H, N, D);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        float gpu_ms = 0.0f;
        for (int rep = 0; rep < kRepeat; ++rep) {
            CHECK_CUDA(cudaEventRecord(s));
            attn_v3::flash_attn_v3_kernel<<<grid,block,smem>>>(
                dQ, dK, dV, dO, B, H, N, D);
            CHECK_CUDA(cudaEventRecord(e));
            CHECK_CUDA(cudaEventSynchronize(e));
            float ms; CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
            gpu_ms += ms;
        }
        gpu_ms /= kRepeat;

        CHECK_CUDA(cudaMemcpy(O_gpu.data(), dO, total*4, cudaMemcpyDeviceToHost));

        double diff = common::MaxAbsDiff(O_cpu, O_gpu);
        bool ok = diff < 1e-3;
        double gflops = 4.0*B*H*N*N*D/(gpu_ms*1e6);

        std::cout << B << "  " << H << "    " << N << "     " << D << "     "
                  << std::fixed << std::setprecision(3) << cpu_ms << "      "
                  << std::setprecision(4) << gpu_ms << "      "
                  << std::setprecision(1) << gflops << "    "
                  << (ok ? "PASS" : "FAIL") << "\n";
        ofs << B<<","<<H<<","<<N<<","<<D<<","<<cpu_ms<<","<<gpu_ms<<","
            << gflops<<","<<diff<<","<<(ok?"PASS":"FAIL")<<"\n";

        CHECK_CUDA(cudaEventDestroy(s));
        CHECK_CUDA(cudaEventDestroy(e));
        CHECK_CUDA(cudaFree(dQ)); CHECK_CUDA(cudaFree(dK));
        CHECK_CUDA(cudaFree(dV)); CHECK_CUDA(cudaFree(dO));
    }
    std::cout << "\nSaved to data/results/flash_attention_v3_results.csv\n";
    return 0;
}
