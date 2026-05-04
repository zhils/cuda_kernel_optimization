// RMSNorm V1: 相对 V0 仅做全局内存两类优化
//
// 1) 减少全局内存读写
//    - V0 每行对 x 遍历两次（先算平方和，再缩放），全局读 x 为 2×cols。
//    - 当单行长度 cols 可在共享内存中暂存整行 x 时：全局只读一遍 x 到共享内存，平方和与写 y 均从共享读，全局读 x 降为 1×cols。
//    - cols 过大时回退为两阶段 stream 路径，仍用 block 内树形归约得到 sq_sum（修正仅单 warp 归约的错误）。
//    - 测试场景，一行最多4096个数据，也就是4K,float 2个字节，也就是8K，符合共享内存限制。
//
// 2) 优化全局内存读写形式
//    - 只读路径对 x、weight 使用 __ldg
//    - cols % 4 == 0 且指针 16 字节对齐时使用 float4

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"
#include "rmsnorm/test_utils.h"

namespace {

static constexpr float kEps = 1e-5f;
constexpr int kThreads = 256;
constexpr std::size_t kMaxDynamicSmemBytes = 40 * 1024;

inline bool CanStageRow(int cols) {
    const std::size_t need =
        (static_cast<std::size_t>(cols) + static_cast<std::size_t>(kThreads)) * sizeof(float);
    return need <= kMaxDynamicSmemBytes;
}

__device__ inline float4 LoadFloat4(const float* p) {
    return __ldg(reinterpret_cast<const float4*>(p));
}

__device__ inline void StoreFloat4(float* p, const float4& v) {
    *reinterpret_cast<float4*>(p) = v;
}

// ---------- Staged: 整行 x 驻留共享内存，全局只读一次 x ----------
__global__ void RMSNormV1StagedKernel(
    const float* __restrict__ x, 
    float* __restrict__ y,
    const float* __restrict__ weight, 
    int rows, int cols,float eps
) {
    extern __shared__ float sdata[];
    float* s_row = sdata;
    float* s_red = sdata + cols;

    const int tid = threadIdx.x;
    const int r = blockIdx.x;
    if (r >= rows) return;

    const float* row_x = x + static_cast<std::size_t>(r) * cols;
    float* row_y = y + static_cast<std::size_t>(r) * cols;

    const bool align4 =
        ((reinterpret_cast<std::uintptr_t>(row_x) % 16u) == 0u) && ((cols % 4) == 0);

    if (align4) {
        int c = tid * 4;
        for (; c + 3 < cols; c += kThreads * 4) {
            StoreFloat4(s_row + c, LoadFloat4(row_x + c));
        }
        for (int k = c; k < cols; ++k) s_row[k] = __ldg(row_x + k);
    } else {
        for (int k = tid; k < cols; k += kThreads) s_row[k] = __ldg(row_x + k);
    }
    __syncthreads();

    float ps = 0.f;
    if (align4) {
        int c = tid * 4;
        for (; c + 3 < cols; c += kThreads * 4) {
            const float4 v = *reinterpret_cast<const float4*>(s_row + c);
            ps += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        for (int k = c; k < cols; ++k) {
            const float v = s_row[k];
            ps += v * v;
        }
    } else {
        for (int k = tid; k < cols; k += kThreads) {
            const float v = s_row[k];
            ps += v * v;
        }
    }
    s_red[tid] = ps;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (tid < s) s_red[tid] += s_red[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        const float sq_sum = s_red[0];
        s_red[0] = rsqrtf(sq_sum / static_cast<float>(cols) + eps);
    }
    __syncthreads();
    const float rms = s_red[0];

    if (align4) {
        int c = tid * 4;
        for (; c + 3 < cols; c += kThreads * 4) {
            const float4 vx = *reinterpret_cast<const float4*>(s_row + c);
            const float4 vw = LoadFloat4(weight + c);
            StoreFloat4(row_y + c, make_float4(vx.x * rms * vw.x, vx.y * rms * vw.y,
                                                vx.z * rms * vw.z, vx.w * rms * vw.w));
        }
        for (int k = c; k < cols; ++k)
            row_y[k] = s_row[k] * rms * __ldg(weight + k);
    } else {
        for (int k = tid; k < cols; k += kThreads)
            row_y[k] = s_row[k] * rms * __ldg(weight + k);
    }
}

// ---------- Stream: 两阶段读 x，__ldg + float4；block 全线程归约 sq_sum ----------
__global__ void RMSNormV1StreamKernel(const float* __restrict__ x, float* __restrict__ y,
                                        const float* __restrict__ weight, int rows, int cols,
                                        float eps) {
    extern __shared__ float s_red[];

    const int tid = threadIdx.x;
    const int r = blockIdx.x;
    if (r >= rows) return;

    const float* row_x = x + static_cast<std::size_t>(r) * cols;
    float* row_y = y + static_cast<std::size_t>(r) * cols;

    const bool align4 =
        ((reinterpret_cast<std::uintptr_t>(row_x) % 16u) == 0u) && ((cols % 4) == 0);

    float ps = 0.f;
    if (align4) {
        int c = tid * 4;
        for (; c + 3 < cols; c += kThreads * 4) {
            const float4 v = LoadFloat4(row_x + c);
            ps += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        for (int k = c; k < cols; ++k) {
            const float v = __ldg(row_x + k);
            ps += v * v;
        }
    } else {
        for (int k = tid; k < cols; k += kThreads) {
            const float v = __ldg(row_x + k);
            ps += v * v;
        }
    }
    s_red[tid] = ps;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (tid < s) s_red[tid] += s_red[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        const float sq_sum = s_red[0];
        s_red[0] = rsqrtf(sq_sum / static_cast<float>(cols) + eps);
    }
    __syncthreads();
    const float rms = s_red[0];

    if (align4) {
        int c = tid * 4;
        for (; c + 3 < cols; c += kThreads * 4) {
            const float4 vx = LoadFloat4(row_x + c);
            const float4 vw = LoadFloat4(weight + c);
            StoreFloat4(row_y + c, make_float4(vx.x * rms * vw.x, vx.y * rms * vw.y,
                                                vx.z * rms * vw.z, vx.w * rms * vw.w));
        }
        for (int k = c; k < cols; ++k)
            row_y[k] = __ldg(row_x + k) * rms * __ldg(weight + k);
    } else {
        for (int k = tid; k < cols; k += kThreads)
            row_y[k] = __ldg(row_x + k) * rms * __ldg(weight + k);
    }
}

}  // namespace

static void RMSNormCPU(const float* x, float* y, const float* weight, int rows, int cols,
                       float eps) {
    for (int r = 0; r < rows; ++r) {
        float sq_sum = 0.f;
        for (int c = 0; c < cols; ++c) {
            float val = x[r * cols + c];
            sq_sum += val * val;
        }
        float rms = 1.f / sqrtf(sq_sum / cols + eps);
        for (int c = 0; c < cols; ++c)
            y[r * cols + c] = x[r * cols + c] * rms * weight[c];
    }
}

int main() {
    constexpr int kRepeat = 10;
    constexpr int kTestCases = 5;
    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/rmsnorm_v1_results.csv");
    ofs << "id,rows,cols,gpu_ms,bandwidth_gb_s,max_abs_diff,check\n";

    for (int i = 0; i < kTestCases; ++i) {
        auto cfg = rmsnorm::RandomTestConfig(2026 + i);
        int rows = cfg.rows, cols = cfg.cols, n = rows * cols;
        std::vector<float> x = rmsnorm::RandomMatrix(rows, cols, 2026 + i);
        std::vector<float> w = rmsnorm::RandomWeight(cols, 2026 + i + 100);
        std::vector<float> cpu(n), gpu(n);
        RMSNormCPU(x.data(), cpu.data(), w.data(), rows, cols, kEps);

        float *dx, *dy, *dw;
        CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dw, cols * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dw, w.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

        const std::size_t smem_staged =
            (static_cast<std::size_t>(cols) + kThreads) * sizeof(float);
        const std::size_t smem_stream = static_cast<std::size_t>(kThreads) * sizeof(float);

        auto dispatch = [&]() {
            if (CanStageRow(cols)) {
                RMSNormV1StagedKernel<<<rows, kThreads, smem_staged>>>(dx, dy, dw, rows, cols,
                                                                       kEps);
            } else {
                RMSNormV1StreamKernel<<<rows, kThreads, smem_stream>>>(dx, dy, dw, rows, cols,
                                                                       kEps);
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

        // 近似：staged 每元素 1×读 x + 1×读 weight + 1×写 y；stream 多一遍读 x
        const double bytes =
            static_cast<double>(n) * sizeof(float) * (CanStageRow(cols) ? 3.0 : 4.0);
        const double bw = bytes / (static_cast<double>(gpu_ms) * 1e6);

        std::cout << rows << "x" << cols << (CanStageRow(cols) ? " [staged]" : " [stream]")
                  << " | " << std::fixed << std::setprecision(4) << gpu_ms << " ms"
                  << " | " << std::setprecision(1) << bw << " GB/s"
                  << " | " << (ok ? "PASS" : "FAIL") << "\n";

        ofs << i << "," << rows << "," << cols << "," << gpu_ms << "," << bw << ","
            << common::MaxAbsDiff(cpu, gpu) << "," << (ok ? "PASS" : "FAIL") << "\n";

        CHECK_CUDA(cudaEventDestroy(s));
        CHECK_CUDA(cudaEventDestroy(e));
        CHECK_CUDA(cudaFree(dx));
        CHECK_CUDA(cudaFree(dy));
        CHECK_CUDA(cudaFree(dw));
    }
    return 0;
}
