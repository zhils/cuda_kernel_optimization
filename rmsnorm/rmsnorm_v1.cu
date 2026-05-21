// RMSNorm V1: 1D block 架构，每个 warp 处理一行。行数据通过共享内存 staging，
// lane0 串行归约平方和，配合 float4 向量化读写减少内存事务。

#include <cuda_runtime.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../common/include/common/benchmark.h"
#include "../common/include/common/cuda_utils.h"
// CPU 参考实现定义见下方

static void RMSNormCPU(const float* x, const float* weight, float* y, int rows, int cols, float eps) {
  for (int i = 0; i < rows; ++i) {
    double ss = 0;
    for (int j = 0; j < cols; ++j) ss += (double)x[i * cols + j] * x[i * cols + j];
    double rms = sqrt(ss / cols + (double)eps);
    for (int j = 0; j < cols; ++j) y[i * cols + j] = x[i * cols + j] / (float)rms * weight[j];
  }
}

#include "test_utils.h"

#define EPS 1e-5f
#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4

__global__ void RMSNormV1Kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    int rows, 
    int cols, 
    float eps
) {
    // 计算 warp 和行索引
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= rows) return;

    const float* row_x = x + row * cols;
    float* row_y = y + row * cols;

    __shared__ float s_sq_sum[WARPS_PER_BLOCK];
    extern __shared__ float s_data[];
    float* s_row = s_data + warp_id * cols;

    // float4 向量化加载数据到共享内存
    const int cols4 = cols / 4;
    for (int c = lane; c < cols4; c += WARP_SIZE) {
        const float4 v = *reinterpret_cast<const float4*>(row_x + c * 4);
        s_row[c * 4 + 0] = v.x;
        s_row[c * 4 + 1] = v.y;
        s_row[c * 4 + 2] = v.z;
        s_row[c * 4 + 3] = v.w;
    }

    // 处理尾项：cols 不是 4 的倍数
    for (int c = cols4 * 4 + lane; c < cols; c += WARP_SIZE) {
        s_row[c] = row_x[c];
    }
    __syncthreads();

    // lane0 串行归约平方和
    if (lane == 0) {
        float sq_sum = 0.f;
        for (int c = 0; c < cols; ++c) {
            const float val = s_row[c];
            sq_sum += val * val;
        }
        s_sq_sum[warp_id] = sq_sum;
    }
    __syncthreads();

    // float4 向量化写回结果
    const float rms = rsqrtf(s_sq_sum[warp_id] / static_cast<float>(cols) + eps);
    for (int c = lane; c < cols4; c += WARP_SIZE) {
        const float4 vx = *reinterpret_cast<const float4*>(s_row + c * 4);
        const float4 vw = *reinterpret_cast<const float4*>(weight + c * 4);
        float4 vy;
        vy.x = vx.x * rms * vw.x;
        vy.y = vx.y * rms * vw.y;
        vy.z = vx.z * rms * vw.z;
        vy.w = vx.w * rms * vw.w;
        *reinterpret_cast<float4*>(row_y + c * 4) = vy;
    }

    // 处理尾项写回
    for (int c = cols4 * 4 + lane; c < cols; c += WARP_SIZE) {
        row_y[c] = s_row[c] * rms * weight[c];
    }
}

int main() {
    constexpr int kRepeat = 10;
    constexpr int kTestCases = 5;
    const std::string results_dir = common::EnsureResultsDir();
    std::ofstream ofs(results_dir + "/rmsnorm_v1_results.csv");
    ofs << "id,rows,cols,gpu_ms,bandwidth_gb_s,max_abs_diff,check\n";

    const std::vector<std::pair<int, int>> test_sizes = {
        {128, 128}, {256, 256}, {512, 512}, {1024, 1024}, {4096, 4096}};

    for (int i = 0; i < kTestCases; ++i) {
        // 确定维度并生成测试数据
        int rows = test_sizes[i].first;
        int cols = test_sizes[i].second;
        int n = rows * cols;
        std::vector<float> x = rmsnorm::RandomMatrix(rows, cols, 2026 + i);
        std::vector<float> w = rmsnorm::RandomWeight(cols, 2026 + i + 100);
        std::vector<float> cpu(n), gpu(n);
        RMSNormCPU(x.data(), w.data(), cpu.data(), rows, cols, EPS);

        // 分配 GPU 内存并拷贝数据到 GPU
        float *dx, *dy, *dw;
        CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dw, cols * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dw, w.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

        // 预热
        const size_t smem_size = WARPS_PER_BLOCK * cols * sizeof(float);
        dim3 block(BLOCK_SIZE);
        dim3 grid((rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

        cudaFuncSetAttribute(RMSNormV1Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        RMSNormV1Kernel<<<grid, block, smem_size>>>(dx, dy, dw, rows, cols, EPS);
        CHECK_CUDA(cudaDeviceSynchronize());

        // 计时循环
        cudaEvent_t s, e;
        CHECK_CUDA(cudaEventCreate(&s));
        CHECK_CUDA(cudaEventCreate(&e));
        CHECK_CUDA(cudaEventRecord(s));
        for (int rep = 0; rep < kRepeat; ++rep) {
            RMSNormV1Kernel<<<grid, block, smem_size>>>(dx, dy, dw, rows, cols, EPS);
        }
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));
        float gpu_ms_total = 0.f;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_total, s, e));
        const float gpu_ms = gpu_ms_total / static_cast<float>(kRepeat);

        // 拷贝结果回 CPU
        CHECK_CUDA(cudaMemcpy(gpu.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));

        // 校验与结果输出
        bool ok = common::CheckEqual(cpu, gpu, 1e-4f);

        const double bytes = static_cast<double>(n) * sizeof(float) * 2.0;
        const double bw = bytes / (static_cast<double>(gpu_ms) * 1e6);

        std::cout << rows << "x" << cols
                  << " | " << std::fixed << std::setprecision(4) << gpu_ms << " ms"
                  << " | " << std::setprecision(1) << bw << " GB/s"
                  << " | " << (ok ? "PASS" : "FAIL") << "\n";

        ofs << i << "," << rows << "," << cols << "," << gpu_ms << "," << bw << ","
            << common::MaxAbsDiff(cpu, gpu) << "," << (ok ? "PASS" : "FAIL") << "\n";

        // 释放资源
        CHECK_CUDA(cudaEventDestroy(s));
        CHECK_CUDA(cudaEventDestroy(e));
        CHECK_CUDA(cudaFree(dx));
        CHECK_CUDA(cudaFree(dy));
        CHECK_CUDA(cudaFree(dw));
    }
    return 0;
}
