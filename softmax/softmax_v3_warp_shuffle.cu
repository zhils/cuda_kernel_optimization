// Softmax V3: Warp shuffle reduction + vectorized memory access
//
// Key optimizations over V2:
//   - __shfl_sync for intra-warp reduction (no shared memory needed)
//   - float4 vectorized loads/stores (4x fewer memory transactions)
//   - Zero shared memory usage → higher occupancy
//   - Suitable for cols <= 128 (single warp per row)

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

__inline__ __device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void SoftmaxWarpVectorizedKernel(const float* __restrict__ x,
                                             float* __restrict__ y,
                                             int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;

    int tid = threadIdx.x;
    int col_base = tid * 4;

    float4 val4;
    if (col_base + 3 < cols) {
        val4 = reinterpret_cast<const float4*>(x + r * cols)[tid];
    } else {
        val4.x = (col_base < cols) ? x[r * cols + col_base] : -INFINITY;
        val4.y = (col_base + 1 < cols) ? x[r * cols + col_base + 1] : -INFINITY;
        val4.z = (col_base + 2 < cols) ? x[r * cols + col_base + 2] : -INFINITY;
        val4.w = (col_base + 3 < cols) ? x[r * cols + col_base + 3] : -INFINITY;
    }

    float max_val = fmaxf(fmaxf(val4.x, val4.y), fmaxf(val4.z, val4.w));
    max_val = warpReduceMax(max_val);
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    val4.x = expf(val4.x - max_val);
    val4.y = expf(val4.y - max_val);
    val4.z = expf(val4.z - max_val);
    val4.w = expf(val4.w - max_val);

    float sum_val = val4.x + val4.y + val4.z + val4.w;
    sum_val = warpReduceSum(sum_val);
    sum_val = __shfl_sync(0xffffffff, sum_val, 0);

    if (col_base + 3 < cols) {
        val4.x /= sum_val;
        val4.y /= sum_val;
        val4.z /= sum_val;
        val4.w /= sum_val;
        reinterpret_cast<float4*>(y + r * cols)[tid] = val4;
    } else {
        if (col_base < cols) y[r * cols + col_base] = val4.x / sum_val;
        if (col_base + 1 < cols) y[r * cols + col_base + 1] = val4.y / sum_val;
        if (col_base + 2 < cols) y[r * cols + col_base + 2] = val4.z / sum_val;
    }
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
        if (cols > 128) continue;

        std::vector<float> x(n), cpu(n), gpu(n);
        common::InitMatrix(x, rows, cols);
        SoftmaxCPU(x.data(), cpu.data(), rows, cols);

        float *dx, *dy;
        CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        int threads = 32;

        SoftmaxWarpVectorizedKernel<<<rows, threads>>>(dx, dy, rows, cols);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        CHECK_CUDA(cudaEventRecord(s));
        for (int rep = 0; rep < kRepeat; ++rep)
            SoftmaxWarpVectorizedKernel<<<rows, threads>>>(dx, dy, rows, cols);
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));
        float gpu_ms_total = 0;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_total, s, e));
        float gpu_ms = gpu_ms_total / kRepeat;

        CHECK_CUDA(cudaMemcpy(gpu.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = common::CheckEqual(cpu, gpu, 1e-4f);

        double bytes = static_cast<double>(n) * sizeof(float) * 2.0;
        double bw = bytes / (gpu_ms * 1e6);

        std::cout << rows << "x" << cols
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
