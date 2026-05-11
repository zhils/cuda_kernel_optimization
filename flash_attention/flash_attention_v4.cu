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
// Flash Attention v4: 提升计算强度
// ============================================================================
// 优化策略（相对 v3）：
//   1. SMEM padding（D_pitch = D + 1）：避免 bank conflict
//   2. __fmaf_rn 替代 a*b+c：指令数减半，提升 FMA 吞吐
//   3. 双路 ILP 累加器：两个独立 dot product 交错计算，隐藏 FMA 延迟
//   4. #pragma unroll 4：展开内层 dot product 循环
//   5. 寄存器缓存 Q 值：减少 SMEM 读取次数
// ============================================================================

namespace attn_v4 {

constexpr int kBr = 32;
constexpr int kBc = 32;
constexpr int kBlockSize = 256;

__global__ void flash_attn_v4_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, 
    int H, 
    int N, 
    int D
){
    int bh = blockIdx.x;
    int q_tile_id = blockIdx.y;
    if (bh >= B * H) return;

    // 使用 D_pitch = D + 1 避免 SMEM bank conflict（32-bank 对齐破坏）
    // 当 D 是 32/64/128 时，stride D 会导致所有线程访问同一 bank
    const int D_pitch = D + 1;

    extern __shared__ float smem[];
    float* Q_smem = smem;                     // [kBr * D_pitch]
    float* K_smem = smem + kBr * D_pitch;     // [kBc * D_pitch]
    float* V_smem = smem + kBr * D_pitch + kBc * D_pitch;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    int q_start = q_tile_id * kBr;
    int q_size = min(kBr, N - q_start);
    int num_kv_tiles = (N + kBc - 1) / kBc;

    int rows_per_warp = kBr / (kBlockSize / 32);
    int my_row_start = warp_id * rows_per_warp;
    int my_row_end = min(my_row_start + rows_per_warp, q_size);

    int d_per_lane = (D + 31) / 32;

    // ------------------------------------------------------------------
    // Phase 1: 加载 Q tile → SMEM（使用 D_pitch 行跨度避免 bank conflict）
    // 每个线程 4 个 float 向量化加载
    // ------------------------------------------------------------------
    {
        int total = kBr * D;
        int idx = tid;
        while (idx < total) {
            int r = idx / D, d = idx % D;
            if (q_start + r < N) {
                Q_smem[r * D_pitch + d] =
                    Q[((size_t)bh * N + q_start + r) * D + d];
            }
            idx += kBlockSize;
        }
    }
    __syncthreads();

    // 寄存器缓存当前 warp 负责的 Q 行（减少 SMEM 读取延迟）
    // q_reg[ri][k] 存储第 ri 行的第 k 个 d_per_lane 元素
    float q_reg[4][4] = {{0.0f}};
    for (int ri = 0; ri < rows_per_warp; ++ri) {
        int qr = my_row_start + ri;
        if (qr >= q_size) continue;
        #pragma unroll
        for (int k = 0; k < d_per_lane; ++k) {
            q_reg[ri][k] = Q_smem[qr * D_pitch + lane * d_per_lane + k];
        }
    }

    float o_acc[4][4] = {{0.0f}};
    float m_prev[4], ell[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        m_prev[i] = -INFINITY;
        ell[i] = 0.0f;
    }

    // ------------------------------------------------------------------
    // Phase 2: 遍历 KV tiles
    // ------------------------------------------------------------------
    for (int kv = 0; kv < num_kv_tiles; ++kv) {
        int kv_start = kv * kBc;
        int kv_size = min(kBc, N - kv_start);

        // 加载 K/V tile 到 SMEM
        {
            int total = kBc * D;
            int idx = tid;
            while (idx < total) {
                int r = idx / D, d = idx % D;
                if (kv_start + r < N) {
                    K_smem[r * D_pitch + d] =
                        K[((size_t)bh * N + kv_start + r) * D + d];
                    V_smem[r * D_pitch + d] =
                        V[((size_t)bh * N + kv_start + r) * D + d];
                }
                idx += kBlockSize;
            }
        }
        __syncthreads();

        for (int ri = 0; ri < rows_per_warp; ++ri) {
            int qr = my_row_start + ri;
            if (qr >= q_size) continue;

            // ----------------------------------------------------------
            // S_row = Q_row @ K_tile^T
            // 双路 ILP：两个独立的累加器交叠，隐藏 FMA 延迟
            // 使用 __fmaf_rn 替代乘法+加法
            // #pragma unroll 展开内层循环
            // ----------------------------------------------------------
            float S_row[32];
            #pragma unroll
            for (int c = 0; c < kv_size; ++c) {
                float sum0 = 0.0f, sum1 = 0.0f;
                int d_base = lane * d_per_lane;

                // 双路 ILP：两个累加器交替 FMA
                int k = 0;
                for (; k + 1 < d_per_lane; k += 2) {
                    sum0 = __fmaf_rn(q_reg[ri][k],
                        K_smem[c * D_pitch + d_base + k], sum0);
                    sum1 = __fmaf_rn(q_reg[ri][k + 1],
                        K_smem[c * D_pitch + d_base + k + 1], sum1);
                }
                if (k < d_per_lane) {
                    sum0 = __fmaf_rn(q_reg[ri][k],
                        K_smem[c * D_pitch + d_base + k], sum0);
                }
                float sum = sum0 + sum1;

                // warp shuffle 归约
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    sum += __shfl_xor_sync(0xffffffff, sum, offset);
                S_row[c] = __shfl_sync(0xffffffff, sum, 0);
            }

            // Online softmax rescaling
            float row_max = -INFINITY;
            #pragma unroll
            for (int c = 0; c < kv_size; ++c)
                row_max = fmaxf(row_max, S_row[c]);
            float m_new = fmaxf(m_prev[ri], row_max);
            float old_scale = expf(m_prev[ri] - m_new);

            // O_accum 更新（双路 ILP）
            #pragma unroll
            for (int k = 0; k < d_per_lane; ++k) {
                float new_contrib = 0.0f;
                // 内层循环：exp(S[c]-m_new) * V 求和
                int d_idx = lane * d_per_lane + k;
                float sum0 = 0.0f, sum1 = 0.0f;
                int c = 0;
                for (; c + 1 < kv_size; c += 2) {
                    sum0 = __fmaf_rn(expf(S_row[c] - m_new),
                               V_smem[c * D_pitch + d_idx], sum0);
                    sum1 = __fmaf_rn(expf(S_row[c + 1] - m_new),
                               V_smem[(c + 1) * D_pitch + d_idx], sum1);
                }
                for (; c < kv_size; ++c) {
                    sum0 += expf(S_row[c] - m_new) * V_smem[c * D_pitch + d_idx];
                }
                new_contrib = sum0 + sum1;
                o_acc[ri][k] = o_acc[ri][k] * old_scale + new_contrib;
            }

            if (lane == 0) {
                float new_sum = 0.0f;
                #pragma unroll
                for (int c = 0; c < kv_size; ++c)
                    new_sum += expf(S_row[c] - m_new);
                ell[ri] = ell[ri] * old_scale + new_sum;
            }
            m_prev[ri] = m_new;
        }
        __syncthreads();
    }

    // ------------------------------------------------------------------
    // Phase 3: 归一化写回
    // ------------------------------------------------------------------
    for (int ri = 0; ri < rows_per_warp; ++ri) {
        int qr = my_row_start + ri;
        if (qr >= q_size) continue;
        float inv_ell = __shfl_sync(0xffffffff, 1.0f / ell[ri], 0);
        float* o_ptr = O + ((size_t)bh * N + q_start + qr) * D;
        #pragma unroll
        for (int k = 0; k < d_per_lane; ++k) {
            int d_idx = lane * d_per_lane + k;
            if (d_idx < D) {
                o_ptr[d_idx] = o_acc[ri][k] * inv_ell;
            }
        }
    }
}

} // namespace attn_v4

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

int main() {
    constexpr int kWarmup = 1, kRepeat = 10;
    std::vector<std::tuple<int,int,int,int>> cases = {
        {1,1,64,32},{1,1,128,64},{1,2,256,64},{1,4,512,64},{1,8,1024,32}
    };

    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/flash_attention_v4_results.csv");
    ofs << "B,H,N,D,cpu_ms,gpu_ms,gflops,max_abs_diff,check\n";

    std::cout << "=== Flash Attention V4 (Bank-free SMEM + ILP + __fmaf_rn) ===\n"
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

        int D_pitch = D + 1;
        size_t smem = (attn_v4::kBr + 2*attn_v4::kBc) * D_pitch * sizeof(float);
        int num_q_tiles = (N + attn_v4::kBr - 1) / attn_v4::kBr;
        dim3 grid(B*H, num_q_tiles);
        dim3 block(attn_v4::kBlockSize);

        CHECK_CUDA(cudaFuncSetAttribute(attn_v4::flash_attn_v4_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

        for (int w = 0; w < kWarmup; ++w)
            attn_v4::flash_attn_v4_kernel<<<grid,block,smem>>>(
                dQ, dK, dV, dO, B, H, N, D);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        float gpu_ms = 0.0f;
        for (int rep = 0; rep < kRepeat; ++rep) {
            CHECK_CUDA(cudaEventRecord(s));
            attn_v4::flash_attn_v4_kernel<<<grid,block,smem>>>(
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
    std::cout << "\nSaved to data/results/flash_attention_v4_results.csv\n";
    return 0;
}
