// RMSNorm V3: Based on V2 + auto-select block size by cols.
//
// Keep V2 optimizations:
//   - staged path + stream fallback
//   - alignment-guarded float4 vectorization
//   - warp + cross-warp reduction
//
// V3 adds:
//   - runtime block-size selection by cols
//   - same kernel path with better launch configuration adaptability

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
static constexpr float kEps = 1e-5f;
static constexpr int WARP_SIZE = 32;
static constexpr int kMaxThreads = 512;
static constexpr std::size_t kMaxDynamicSmemBytes = 40 * 1024;
}  // namespace

inline int PickThreadsByCols(int cols) {
    if (cols <= 128) return 64;
    if (cols <= 512) return 128;
    if (cols <= 2048) return 256;
    return 512;
}

inline bool CanStageRow(int cols, int threads) {
    const std::size_t need =
        (static_cast<std::size_t>(cols) + static_cast<std::size_t>(threads)) * sizeof(float);
    return need <= kMaxDynamicSmemBytes;
}

__device__ __forceinline__ float4 LoadFloat4(const float* p) {
    return __ldg(reinterpret_cast<const float4*>(p));
}

__device__ __forceinline__ void StoreFloat4(float* p, const float4& v) {
    *reinterpret_cast<float4*>(p) = v;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float warp_sums[kMaxThreads / WARP_SIZE];
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int warp_count = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    val = warpReduceSum(val);
    if (lane == 0) warp_sums[warp_id] = val;
    __syncthreads();

    float sum = 0.f;
    if (warp_id == 0) {
        sum = (lane < warp_count) ? warp_sums[lane] : 0.f;
        sum = warpReduceSum(sum);
    }
    __syncthreads();
    return sum;
}

__global__ void RMSNormV3StagedKernel(const float* __restrict__ x,
                                      float* __restrict__ y,
                                      const float* __restrict__ weight,
                                      int rows, int cols, float eps) {
    extern __shared__ float sdata[];
    float* s_row = sdata;
    float* s_red = sdata + cols;

    const int tid = threadIdx.x;
    const int r = blockIdx.x;
    if (r >= rows) return;

    const int offset = r * cols;
    const float* row_x = x + static_cast<std::size_t>(offset);
    float* row_y = y + static_cast<std::size_t>(offset);

    const bool align4 =
        (cols % 4 == 0) &&
        (reinterpret_cast<std::uintptr_t>(row_x) % 16u == 0u) &&
        (reinterpret_cast<std::uintptr_t>(row_y) % 16u == 0u) &&
        (reinterpret_cast<std::uintptr_t>(weight) % 16u == 0u);

    if (align4) {
        int c = tid * 4;
        for (; c + 3 < cols; c += blockDim.x * 4) {
            StoreFloat4(s_row + c, LoadFloat4(row_x + c));
        }
        for (int c1 = c; c1 < cols; ++c1) s_row[c1] = __ldg(row_x + c1);
    } else {
        for (int c = tid; c < cols; c += blockDim.x) s_row[c] = __ldg(row_x + c);
    }
    __syncthreads();

    float sq_sum = 0.f;
    if (align4) {
        int c = tid * 4;
        for (; c + 3 < cols; c += blockDim.x * 4) {
            const float4 v = *reinterpret_cast<const float4*>(s_row + c);
            sq_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        for (int c1 = c; c1 < cols; ++c1) {
            const float v = s_row[c1];
            sq_sum += v * v;
        }
    } else {
        for (int c = tid; c < cols; c += blockDim.x) {
            const float v = s_row[c];
            sq_sum += v * v;
        }
    }

    sq_sum = blockReduceSum(sq_sum);
    if (tid == 0) s_red[0] = rsqrtf(sq_sum / static_cast<float>(cols) + eps);
    __syncthreads();
    const float rms = s_red[0];

    if (align4) {
        int c = tid * 4;
        for (; c + 3 < cols; c += blockDim.x * 4) {
            const float4 vx = *reinterpret_cast<const float4*>(s_row + c);
            const float4 vw = LoadFloat4(weight + c);
            StoreFloat4(row_y + c, make_float4(vx.x * rms * vw.x, vx.y * rms * vw.y,
                                               vx.z * rms * vw.z, vx.w * rms * vw.w));
        }
        for (int c1 = c; c1 < cols; ++c1) row_y[c1] = s_row[c1] * rms * __ldg(weight + c1);
    } else {
        for (int c = tid; c < cols; c += blockDim.x) {
            row_y[c] = s_row[c] * rms * __ldg(weight + c);
        }
    }
}

__global__ void RMSNormV3StreamKernel(const float* __restrict__ x,
                                      float* __restrict__ y,
                                      const float* __restrict__ weight,
                                      int rows, int cols, float eps) {
    __shared__ float s_red[kMaxThreads / WARP_SIZE];
    const int tid = threadIdx.x;
    const int r = blockIdx.x;
    if (r >= rows) return;

    const int offset = r * cols;
    const float* row_x = x + static_cast<std::size_t>(offset);
    float* row_y = y + static_cast<std::size_t>(offset);
    const int n4 = cols / 4;

    const bool align4 =
        (cols % 4 == 0) &&
        (reinterpret_cast<std::uintptr_t>(row_x) % 16u == 0u) &&
        (reinterpret_cast<std::uintptr_t>(row_y) % 16u == 0u) &&
        (reinterpret_cast<std::uintptr_t>(weight) % 16u == 0u);

    float sq_sum = 0.f;
    if (align4) {
        for (int c = tid; c < n4; c += blockDim.x) {
            const float4 v = LoadFloat4(row_x + c * 4);
            sq_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (int c = tid; c < cols; c += blockDim.x) {
            const float v = __ldg(row_x + c);
            sq_sum += v * v;
        }
    }

    sq_sum = blockReduceSum(sq_sum);
    if (tid == 0) s_red[0] = rsqrtf(sq_sum / static_cast<float>(cols) + eps);
    __syncthreads();
    const float rms = s_red[0];

    if (align4) {
        for (int c = tid; c < n4; c += blockDim.x) {
            const float4 vx = LoadFloat4(row_x + c * 4);
            const float4 vw = LoadFloat4(weight + c * 4);
            StoreFloat4(row_y + c * 4, make_float4(vx.x * rms * vw.x, vx.y * rms * vw.y,
                                                   vx.z * rms * vw.z, vx.w * rms * vw.w));
        }
    } else {
        for (int c = tid; c < cols; c += blockDim.x) {
            row_y[c] = __ldg(row_x + c) * rms * __ldg(weight + c);
        }
    }
}

static void RMSNormCPU(const float* x, float* y, const float* weight,
                       int rows, int cols, float eps) {
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
    auto cases = common::LoadOrCreateTestCasesCsv("data/rmsnorm/test_cases.csv");
    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/rmsnorm_v3_results.csv");
    ofs << "id,rows,cols,threads,gpu_ms,bandwidth_gb_s,max_abs_diff,check\n";

    for (size_t i = 0; i < cases.size(); ++i) {
        int rows = cases[i].rows, cols = cases[i].cols, n = rows * cols;
        std::vector<float> x(n), w(cols), cpu(n), gpu(n);
        common::InitMatrix(x, rows, cols);
        common::InitMatrix(w, 1, cols);
        RMSNormCPU(x.data(), cpu.data(), w.data(), rows, cols, kEps);

        float *dx, *dy, *dw;
        CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dw, cols * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dw, w.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

        const int threads = PickThreadsByCols(cols);
        const std::size_t smem_staged =
            (static_cast<std::size_t>(cols) + static_cast<std::size_t>(threads)) * sizeof(float);
        const bool use_staged = CanStageRow(cols, threads);
        auto dispatch = [&]() {
            if (use_staged) {
                RMSNormV3StagedKernel<<<rows, threads, smem_staged>>>(dx, dy, dw, rows, cols,
                                                                      kEps);
            } else {
                RMSNormV3StreamKernel<<<rows, threads>>>(dx, dy, dw, rows, cols, kEps);
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

        const double bytes =
            static_cast<double>(n) * sizeof(float) * (use_staged ? 3.0 : 4.0);
        const double bw = bytes / (static_cast<double>(gpu_ms) * 1e6);

        std::cout << rows << "x" << cols << " [threads=" << threads << "]"
                  << (use_staged ? " [staged]" : " [stream]")
                  << " | " << std::fixed << std::setprecision(4) << gpu_ms << " ms"
                  << " | " << std::setprecision(1) << bw << " GB/s"
                  << " | " << (ok ? "PASS" : "FAIL") << "\n";

        ofs << i << "," << rows << "," << cols << "," << threads << ","
            << gpu_ms << "," << bw << ","
            << common::MaxAbsDiff(cpu, gpu) << "," << (ok ? "PASS" : "FAIL") << "\n";

        CHECK_CUDA(cudaEventDestroy(s));
        CHECK_CUDA(cudaEventDestroy(e));
        CHECK_CUDA(cudaFree(dx));
        CHECK_CUDA(cudaFree(dy));
        CHECK_CUDA(cudaFree(dw));
    }
    return 0;
}
