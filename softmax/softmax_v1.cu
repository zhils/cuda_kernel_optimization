// Softmax V1: 相对 V0 仅做全局内存层面的两类优化
//
// 1) 减少全局内存读写
//    - 当单行长度 cols 可在共享内存中暂存整行 exp 时：去掉「先写 y=exp，再读 y 做归一化」
//      的中间全局往返，每个元素对 y 从「写+读+写」变为「写一次」。
//    - cols 较大、无法暂存时：仍为三阶段，但读 x 用 __ldg、在可对齐时用 float4，减少无效带宽与指令数。
//
// 2) 优化全局内存读写形式
//    - 只读路径使用 __ldg（只读缓存）
//    - cols % 4 == 0 且地址 16 字节对齐时使用 float4 合并访问

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

namespace {

constexpr int kThreads = 256;

// 动态共享内存上限（保守，避免超过默认 48KB / 每 block 配置）
constexpr std::size_t kMaxDynamicSmemBytes = 40 * 1024;

inline bool CanStageExp(int cols) {
    const std::size_t need =
        (static_cast<std::size_t>(cols) + static_cast<std::size_t>(kThreads)) * sizeof(float);
    return need <= kMaxDynamicSmemBytes;
}

// ---------- 向量化辅助：仅在 (row*cols + offset) 对齐到 16 字节时使用 float4 ----------
__device__ inline void VectorMax4(const float* __restrict__ base, int c0, float& acc) {
    const float4 v = __ldg(reinterpret_cast<const float4*>(base + c0));
    acc = fmaxf(acc, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
}

__device__ inline float4 LoadExp4(const float* __restrict__ base, float row_max) {
    const float4 vx = __ldg(reinterpret_cast<const float4*>(base));
    return make_float4(expf(vx.x - row_max), expf(vx.y - row_max), expf(vx.z - row_max),
                       expf(vx.w - row_max));
}

__device__ inline float Sum4(const float4& v) { return v.x + v.y + v.z + v.w; }

// 可暂存 exp：max → 写共享 exp → 归约 sum → 一次写全局 y
__global__ void SoftmaxV1StagedKernel(const float* __restrict__ x, float* __restrict__ y,
                                      int rows, int cols) {
    extern __shared__ float sdata[];
    float* s_exp = sdata + blockDim.x;

    const int r = blockIdx.x;
    if (r >= rows) return;

    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const float* row_x = x + static_cast<std::size_t>(r) * cols;

    const bool align4 = ((reinterpret_cast<std::uintptr_t>(row_x) % 16u) == 0u) && ((cols % 4) == 0);

    // ---- max ----
    float tmax = -INFINITY;
    if (align4) {
        int c = tid * 4;
        for (; c + 3 < cols; c += bdim * 4) {
            VectorMax4(row_x, c, tmax);
        }
        for (int k = c; k < cols; ++k) tmax = fmaxf(tmax, __ldg(row_x + k));
    } else {
        for (int k = tid; k < cols; k += bdim) tmax = fmaxf(tmax, __ldg(row_x + k));
    }

    sdata[tid] = tmax;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    const float row_max = sdata[0];

    // ---- exp 写入共享内存，并累计线程局部和 ----
    float tsum = 0.f;
    if (align4) {
        int c = tid * 4;
        for (; c + 3 < cols; c += bdim * 4) {
            const float4 e = LoadExp4(row_x + c, row_max);
            *reinterpret_cast<float4*>(s_exp + c) = e;
            tsum += Sum4(e);
        }
        for (int k = c; k < cols; ++k) {
            const float v = expf(__ldg(row_x + k) - row_max);
            s_exp[k] = v;
            tsum += v;
        }
    } else {
        for (int k = tid; k < cols; k += bdim) {
            const float v = expf(__ldg(row_x + k) - row_max);
            s_exp[k] = v;
            tsum += v;
        }
    }

    sdata[tid] = tsum;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float row_sum = sdata[0];

    // ---- 一次写全局 y（从共享读，无额外全局读）----
    float* row_y = y + static_cast<std::size_t>(r) * cols;
    if (align4) {
        const float inv = 1.f / row_sum;
        int c = tid * 4;
        for (; c + 3 < cols; c += bdim * 4) {
            const float4 e = *reinterpret_cast<const float4*>(s_exp + c);
            const float4 out = make_float4(e.x * inv, e.y * inv, e.z * inv, e.w * inv);
            *reinterpret_cast<float4*>(row_y + c) = out;
        }
        for (int k = c; k < cols; ++k) row_y[k] = s_exp[k] * inv;
    } else {
        const float inv = 1.f / row_sum;
        for (int k = tid; k < cols; k += bdim) row_y[k] = s_exp[k] * inv;
    }
}

// 无法暂存整行：三阶段，但对 x 使用 __ldg + 可选 float4（与 V0 相比减少 y 的中间读仍做不到，仅优化访存形式）
__global__ void SoftmaxV1StreamKernel(const float* __restrict__ x, float* __restrict__ y, int rows,
                                      int cols) {
    extern __shared__ float sdata[];

    const int r = blockIdx.x;
    if (r >= rows) return;

    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const float* row_x = x + static_cast<std::size_t>(r) * cols;
    float* row_y = y + static_cast<std::size_t>(r) * cols;

    const bool align4 = ((reinterpret_cast<std::uintptr_t>(row_x) % 16u) == 0u) && ((cols % 4) == 0);

    float tmax = -INFINITY;
    if (align4) {
        int c = tid * 4;
        for (; c + 3 < cols; c += bdim * 4) {
            VectorMax4(row_x, c, tmax);
        }
        for (int k = c; k < cols; ++k) tmax = fmaxf(tmax, __ldg(row_x + k));
    } else {
        for (int k = tid; k < cols; k += bdim) tmax = fmaxf(tmax, __ldg(row_x + k));
    }

    sdata[tid] = tmax;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    const float row_max = sdata[0];

    float tsum = 0.f;
    if (align4) {
        int c = tid * 4;
        for (; c + 3 < cols; c += bdim * 4) {
            const float4 e = LoadExp4(row_x + c, row_max);
            *reinterpret_cast<float4*>(row_y + c) = e;
            tsum += Sum4(e);
        }
        for (int k = c; k < cols; ++k) {
            const float v = expf(__ldg(row_x + k) - row_max);
            row_y[k] = v;
            tsum += v;
        }
    } else {
        for (int k = tid; k < cols; k += bdim) {
            const float v = expf(__ldg(row_x + k) - row_max);
            row_y[k] = v;
            tsum += v;
        }
    }

    sdata[tid] = tsum;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float row_sum = sdata[0];

    if (align4) {
        const float inv = 1.f / row_sum;
        int c = tid * 4;
        for (; c + 3 < cols; c += bdim * 4) {
            const float4 v = __ldg(reinterpret_cast<const float4*>(row_y + c));
            *reinterpret_cast<float4*>(row_y + c) =
                make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
        }
        for (int k = c; k < cols; ++k) row_y[k] = __ldg(row_y + k) * inv;
    } else {
        const float inv = 1.f / row_sum;
        for (int k = tid; k < cols; k += bdim) row_y[k] = __ldg(row_y + k) * inv;
    }
}

}  // namespace

static void SoftmaxCPU(const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float maxv = x[r * cols];
        for (int c = 1; c < cols; ++c) maxv = std::max(maxv, x[r * cols + c]);
        float sum = 0.f;
        for (int c = 0; c < cols; ++c) {
            float v = std::exp(x[r * cols + c] - maxv);
            y[r * cols + c] = v;
            sum += v;
        }
        for (int c = 0; c < cols; ++c) y[r * cols + c] /= sum;
    }
}

int main() {
    constexpr int kRepeat = 10;
    auto cases = common::LoadOrCreateTestCasesCsv("data/softmax/test_cases.csv");
    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/softmax_v1_results.csv");
    ofs << "id,rows,cols,gpu_ms,bandwidth_gb_s,max_abs_diff,check\n";

    for (size_t i = 0; i < cases.size(); ++i) {
        int rows = cases[i].rows, cols = cases[i].cols, n = rows * cols;
        std::vector<float> x(n), cpu(n), gpu(n);
        common::InitMatrix(x, rows, cols);
        SoftmaxCPU(x.data(), cpu.data(), rows, cols);

        float *dx, *dy;
        CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        const std::size_t smem_staged =
            (static_cast<std::size_t>(cols) + kThreads) * sizeof(float);
        const std::size_t smem_stream = static_cast<std::size_t>(kThreads) * sizeof(float);

        auto dispatch = [&]() {
            if (CanStageExp(cols)) {
                SoftmaxV1StagedKernel<<<rows, kThreads, smem_staged>>>(dx, dy, rows, cols);
            } else {
                SoftmaxV1StreamKernel<<<rows, kThreads, smem_stream>>>(dx, dy, rows, cols);
            }
        };

        dispatch();
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        CHECK_CUDA(cudaEventRecord(s));
        for (int rep = 0; rep < kRepeat; ++rep) dispatch();
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));
        float gpu_ms_total = 0.f;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_total, s, e));
        const float gpu_ms = gpu_ms_total / static_cast<float>(kRepeat);

        CHECK_CUDA(cudaMemcpy(gpu.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = common::CheckEqual(cpu, gpu, 1e-4f);

        // 有效字节量（近似）：读 x 2 遍 + 写 y 1 遍；暂存路径无对 y 的中间读
        const double bytes =
            static_cast<double>(n) * sizeof(float) * (CanStageExp(cols) ? 3.0 : 5.0);
        const double bw = bytes / (static_cast<double>(gpu_ms) * 1e6);

        std::cout << rows << "x" << cols << (CanStageExp(cols) ? " [staged]" : " [stream]")
                  << " | " << std::fixed << std::setprecision(4) << gpu_ms << " ms"
                  << " | " << std::setprecision(1) << bw << " GB/s"
                  << " | " << (ok ? "PASS" : "FAIL") << "\n";

        ofs << i << "," << rows << "," << cols << "," << gpu_ms << "," << bw << ","
            << common::MaxAbsDiff(cpu, gpu) << "," << (ok ? "PASS" : "FAIL") << "\n";

        CHECK_CUDA(cudaEventDestroy(s));
        CHECK_CUDA(cudaEventDestroy(e));
        CHECK_CUDA(cudaFree(dx));
        CHECK_CUDA(cudaFree(dy));
    }
    return 0;
}
