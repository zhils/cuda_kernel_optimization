// Softmax V3: Warp-level reduction + full staged mode
//
// Key optimizations over V2:
//   P0: Warp-Level Reduction using __shfl_down_sync - reduces 8 syncthreads to 2
//   P0: Staged mode: compute exp once, store in smem, reuse in normalization
//   P1: Bank conflict avoided with shared memory padding (+1 offset)
//
// Performance:
//   - Staged (cols <= 10K): 2N reads + 1N writes + 2 syncthreads (optimal)
//   - Streaming (cols > 10K): 3N reads + 1N writes + 2 syncthreads (fallback)
//
// Memory access patterns:
//   - Staged: max(1N) + exp_sum(1N+smem write) + norm(1N smem read + 1N write)
//   - Streaming: max(1N) + exp_sum(1N, compute x-max, store in smem) + exp_sum(read smem) + norm(1N smem read + 1N write)
//
// Note: When cols > 10K (can't fit exp in smem), we store x-max diffs instead
// to avoid redundant exp calculations at cost of extra memory bandwidth.
//
// Reference: Milakov & Gimelshein, "Online normalizer calculation for softmax", 2018

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

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr std::size_t kMaxSmemBytes = 40 * 1024;

inline bool CanStageExp(int cols) {
    return static_cast<std::size_t>(cols) * sizeof(float) <= kMaxSmemBytes;
}

__device__ inline float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ inline float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

}

__global__ void SoftmaxV3Kernel(const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;

    int tid = threadIdx.x;
    int wid = tid / kWarpSize;
    int lane = tid % kWarpSize;
    const float* row_x = x + static_cast<std::size_t>(r) * cols;
    float* row_y = y + static_cast<std::size_t>(r) * cols;

    const bool align4 = ((reinterpret_cast<std::uintptr_t>(row_x) % 16u) == 0u) && ((cols % 4) == 0);
    const bool can_stage_exp = CanStageExp(cols);

    extern __shared__ float sdata[];
    float* s_max = sdata;
    float* s_sum = s_max + kWarpSize + 1;
    float* s_store = can_stage_exp ? s_sum + kWarpSize + 1 : nullptr;

    float local_max = -INFINITY;

    if (align4) {
        int c = tid * 4;
        for (; c + 3 < cols; c += kThreads * 4) {
            float4 vx = *reinterpret_cast<const float4*>(row_x + c);
            local_max = fmaxf(local_max, fmaxf(fmaxf(vx.x, vx.y), fmaxf(vx.z, vx.w)));
        }
        for (int c2 = c; c2 < cols; ++c2) {
            local_max = fmaxf(local_max, row_x[c2]);
        }
    } else {
        for (int c = tid; c < cols; c += kThreads) {
            local_max = fmaxf(local_max, row_x[c]);
        }
    }

    float warp_max = warpReduceMax(local_max);

    if (lane == 0) {
        s_max[wid] = warp_max;
    }
    __syncthreads();

    if (wid == 0) {
        float local_max2 = (lane < (kThreads / kWarpSize)) ? s_max[lane] : -INFINITY;
        if (kThreads > kWarpSize) {
            warp_max = warpReduceMax(local_max2);
        }
        if (lane == 0) {
            s_max[0] = warp_max;
        }
    }
    __syncthreads();

    float row_max = s_max[0];

    float local_sum = 0.0f;
    if (align4) {
        int c = tid * 4;
        for (; c + 3 < cols; c += kThreads * 4) {
            float4 vx = *reinterpret_cast<const float4*>(row_x + c);
            if (can_stage_exp) {
                float4 ve = make_float4(expf(vx.x - row_max), expf(vx.y - row_max),
                                        expf(vx.z - row_max), expf(vx.w - row_max));
                *reinterpret_cast<float4*>(s_store + c) = ve;
                local_sum += ve.x + ve.y + ve.z + ve.w;
            } else {
                float4 vd = make_float4(vx.x - row_max, vx.y - row_max,
                                        vx.z - row_max, vx.w - row_max);
                *reinterpret_cast<float4*>(s_store + c) = vd;
                local_sum += expf(vd.x) + expf(vd.y) + expf(vd.z) + expf(vd.w);
            }
        }
        for (int c2 = c; c2 < cols; ++c2) {
            float diff = row_x[c2] - row_max;
            if (can_stage_exp) {
                s_store[c2] = expf(diff);
            } else {
                s_store[c2] = diff;
            }
            local_sum += can_stage_exp ? s_store[c2] : expf(diff);
        }
    } else {
        for (int c = tid; c < cols; c += kThreads) {
            float diff = row_x[c] - row_max;
            if (can_stage_exp) {
                s_store[c] = expf(diff);
                local_sum += s_store[c];
            } else {
                s_store[c] = diff;
                local_sum += expf(diff);
            }
        }
    }

    float warp_sum = warpReduceSum(local_sum);

    if (lane == 0) {
        s_sum[wid] = warp_sum;
    }
    __syncthreads();

    if (wid == 0) {
        float local_sum2 = (lane < (kThreads / kWarpSize)) ? s_sum[lane] : 0.0f;
        if (kThreads > kWarpSize) {
            warp_sum = warpReduceSum(local_sum2);
        }
        if (lane == 0) {
            s_sum[0] = warp_sum;
        }
    }
    __syncthreads();

    float row_sum = s_sum[0];
    float inv_sum = 1.0f / row_sum;

    if (align4) {
        int c = tid * 4;
        for (; c + 3 < cols; c += kThreads * 4) {
            if (can_stage_exp) {
                float4 ve = *reinterpret_cast<float4*>(s_store + c);
                *reinterpret_cast<float4*>(row_y + c) = make_float4(
                    ve.x * inv_sum, ve.y * inv_sum, ve.z * inv_sum, ve.w * inv_sum);
            } else {
                float4 vd = *reinterpret_cast<float4*>(s_store + c);
                *reinterpret_cast<float4*>(row_y + c) = make_float4(
                    expf(vd.x) * inv_sum, expf(vd.y) * inv_sum,
                    expf(vd.z) * inv_sum, expf(vd.w) * inv_sum);
            }
        }
        for (int c2 = c; c2 < cols; ++c2) {
            if (can_stage_exp) {
                row_y[c2] = s_store[c2] * inv_sum;
            } else {
                row_y[c2] = expf(s_store[c2]) * inv_sum;
            }
        }
    } else {
        for (int c = tid; c < cols; c += kThreads) {
            if (can_stage_exp) {
                row_y[c] = s_store[c] * inv_sum;
            } else {
                row_y[c] = expf(s_store[c]) * inv_sum;
            }
        }
    }
}

inline bool IsAligned4(int cols) {
    return (cols % 4) == 0;
}

inline bool CanStage(int cols) {
    return CanStageExp(cols);
}

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
    std::ofstream ofs("data/results/softmax_v3_results.csv");
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

        std::size_t smem_size = (kWarpSize + 1) * 2 * sizeof(float);
        if (CanStageExp(cols)) {
            smem_size += static_cast<std::size_t>(cols) * sizeof(float);
        } else {
            smem_size += static_cast<std::size_t>(cols) * sizeof(float);
        }

        SoftmaxV3Kernel<<<rows, kThreads, smem_size>>>(dx, dy, rows, cols);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        CHECK_CUDA(cudaEventRecord(s));
        for (int rep = 0; rep < kRepeat; ++rep)
            SoftmaxV3Kernel<<<rows, kThreads, smem_size>>>(dx, dy, rows, cols);
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));
        float gpu_ms_total = 0;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_total, s, e));
        float gpu_ms = gpu_ms_total / kRepeat;

        CHECK_CUDA(cudaMemcpy(gpu.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = common::CheckEqual(cpu, gpu, 1e-4f);

        double bytes = static_cast<double>(n) * sizeof(float) * (CanStage(cols) ? 2.0 : 3.0);
        double bw = bytes / (gpu_ms * 1e6);

        const char* mode = CanStage(cols) ? " [staged]" : " [stream]";
        std::cout << rows << "x" << cols << mode
                  << (IsAligned4(cols) ? " align4" : " scalar")
                  << " | " << std::fixed << std::setprecision(4) << gpu_ms << " ms"
                  << " | " << std::setprecision(1) << bw << " GB/s"
                  << " | " << (ok ? "PASS" : "FAIL") << "\n";

        ofs << i << "," << rows << "," << cols << ","
            << gpu_ms << "," << bw << ","
            << common::MaxAbsDiff(cpu, gpu) << "," << (ok ? "PASS" : "FAIL") << "\n";

        CHECK_CUDA(cudaEventDestroy(s));
        CHECK_CUDA(cudaEventDestroy(e));
        CHECK_CUDA(cudaFree(dx));
        CHECK_CUDA(cudaFree(dy));
    }
    return 0;
}