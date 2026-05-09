// RMSNorm CUB baseline: 使用 cub::BlockReduce 做平方和归约
// 每 block 处理一行，融合归一化，float4 向量化加载/写回
// 目的：与手写 kernel (v1~v3) 对比 CUB 库的开销

#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>
#include <vector>

#include "../common/include/common/benchmark.h"
#include "../common/include/common/cuda_utils.h"
#include "test_utils.h"

#define EPS 1e-5f

// 一个 block 处理一行，grid = rows
constexpr int kBlockSize = 256;

__global__ void RMSNormCUBKernel(const float* __restrict__ x, float* __restrict__ y,
                                  const float* __restrict__ weight, int rows, int cols, float eps) {
  int row = blockIdx.x;
  if (row >= rows) return;
  const float* row_x = x + static_cast<size_t>(row) * cols;
  float* row_y = y + static_cast<size_t>(row) * cols;

  // 每线程累加局部平方和
  float local_sum = 0.0f;
  const int cols4 = cols / 4;
  // 对齐时用 float4
  const bool align4 = (cols % 4 == 0) &&
                      (reinterpret_cast<uintptr_t>(row_x) % 16 == 0) &&
                      (reinterpret_cast<uintptr_t>(row_y) % 16 == 0);
  if (align4) {
    for (int c = threadIdx.x; c < cols4; c += blockDim.x) {
      float4 v = reinterpret_cast<const float4*>(row_x)[c];
      local_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
  } else {
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
      float val = row_x[c];
      local_sum += val * val;
    }
  }

  // cub::BlockReduce 做 warp 间归约
  typedef cub::BlockReduce<float, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp;
  float sq_sum = BlockReduce(temp).Sum(local_sum);

  // 广播 rms 到所有线程
  __shared__ float s_rms;
  if (threadIdx.x == 0) {
    s_rms = rsqrtf(sq_sum / static_cast<float>(cols) + eps);
  }
  __syncthreads();
  float rms = s_rms;

  // 融合归一化写回
  if (align4) {
    for (int c = threadIdx.x; c < cols4; c += blockDim.x) {
      float4 vx = reinterpret_cast<const float4*>(row_x)[c];
      float4 vw = reinterpret_cast<const float4*>(weight)[c];
      reinterpret_cast<float4*>(row_y)[c] =
          make_float4(vx.x * rms * vw.x, vx.y * rms * vw.y, vx.z * rms * vw.z, vx.w * rms * vw.w);
    }
  } else {
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
      row_y[c] = row_x[c] * rms * weight[c];
    }
  }
}

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

#ifdef _WIN32
  _mkdir("data");
  _mkdir("data\\results");
#else
  mkdir("data", 0755);
  mkdir("data/results", 0755);
#endif
  std::ofstream ofs("data/results/rmsnorm_cub_ref_results.csv");
  ofs << "id,rows,cols,gpu_ms,bandwidth_gb_s,max_abs_diff,check\n";

  std::vector<std::pair<int, int>> test_sizes = {
      {128, 128}, {256, 256}, {512, 512}, {1024, 1024}, {4096, 4096}};

  int max_smem = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock, 0));

  for (int i = 0; i < kTestCases; ++i) {
    int rows = test_sizes[i].first;
    int cols = test_sizes[i].second;
    int n = rows * cols;
    std::vector<float> x = rmsnorm::RandomMatrix(rows, cols, 2026 + i);
    std::vector<float> w = rmsnorm::RandomWeight(cols, 2026 + i + 100);
    std::vector<float> cpu(n), gpu(n);
    RMSNormCPU(x.data(), cpu.data(), w.data(), rows, cols, EPS);

    float *dx, *dy, *dw;
    CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dw, cols * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw, w.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(kBlockSize);
    dim3 grid(rows);

    // warmup
    RMSNormCUBKernel<<<grid, block>>>(dx, dy, dw, rows, cols, EPS);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(s));
    for (int rep = 0; rep < kRepeat; ++rep) {
      RMSNormCUBKernel<<<grid, block>>>(dx, dy, dw, rows, cols, EPS);
    }
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    float gpu_ms_total = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_total, s, e));
    const float gpu_ms = gpu_ms_total / static_cast<float>(kRepeat);

    CHECK_CUDA(cudaMemcpy(gpu.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = common::CheckEqual(cpu, gpu, 1e-4f);

    const double bytes = static_cast<double>(n) * sizeof(float) * 2.0;
    const double bw = bytes / (static_cast<double>(gpu_ms) * 1e6);

    std::cout << rows << "x" << cols
              << " | BlockSize=" << kBlockSize
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
