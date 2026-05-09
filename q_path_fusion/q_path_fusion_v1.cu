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

// 用固定模式初始化向量，保证每次 benchmark 可复现。
static void InitVector(std::vector<float>& vec) {
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = static_cast<float>((i * 17 + 13) % 1000) * 0.001f;
  }
}

constexpr int QPF_BLOCK_SIZE = 256;

__global__ void QPathFusionV1Kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ wq,
    const float* __restrict__ bq,
    float* __restrict__ q,
    int rows,
    int cols,
    float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int tx = threadIdx.x;

  // v1 仍是单 kernel：RMSNorm + Linear
  extern __shared__ float s_norm[];
  __shared__ float s_reduce[QPF_BLOCK_SIZE];
  __shared__ float s_inv_rms;

  const float* x_row = x + static_cast<size_t>(row) * cols;
  float* q_row = q + static_cast<size_t>(row) * cols;

  // 1) 每个线程先累计自己负责的平方和
  float part = 0.0f;
  for (int c = tx; c < cols; c += blockDim.x) {
    float v = x_row[c];
    part += v * v;
  }
  s_reduce[tx] = part;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tx < stride) s_reduce[tx] += s_reduce[tx + stride];
    __syncthreads();
  }
  if (tx == 0) {
    s_inv_rms = rsqrtf(s_reduce[0] / static_cast<float>(cols) + eps);
  }
  __syncthreads();

  // 2) 写入归一化后的临时向量
  const float inv = s_inv_rms;
  for (int c = tx; c < cols; c += blockDim.x) {
    s_norm[c] = x_row[c] * inv * gamma[c];
  }
  __syncthreads();

  // 3) 输出列按线程条带分配，减少写回冲突
  for (int out_col = tx; out_col < cols; out_col += blockDim.x) {
    float acc = bq[out_col];
    for (int k = 0; k < cols; ++k) {
      acc += s_norm[k] * wq[static_cast<size_t>(k) * cols + out_col];
    }
    q_row[out_col] = acc;
  }
}

static void QPathFusionCPU(
    const float* x,
    const float* gamma,
    const float* wq,
    const float* bq,
    float* q,
    int rows,
    int cols,
    float eps) {
  std::vector<float> norm(cols, 0.0f);
  for (int r = 0; r < rows; ++r) {
    const float* row_x = x + static_cast<size_t>(r) * static_cast<size_t>(cols);
    float* row_q = q + static_cast<size_t>(r) * static_cast<size_t>(cols);

    float sq_sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
      sq_sum += row_x[c] * row_x[c];
    }
    const float inv_rms = 1.0f / std::sqrt(sq_sum / static_cast<float>(cols) + eps);
    for (int c = 0; c < cols; ++c) {
      norm[c] = row_x[c] * inv_rms * gamma[c];
    }

    for (int out_col = 0; out_col < cols; ++out_col) {
      float acc = bq[out_col];
      for (int k = 0; k < cols; ++k) {
        acc += norm[k] * wq[static_cast<size_t>(k) * static_cast<size_t>(cols) + out_col];
      }
      row_q[out_col] = acc;
    }
  }
}

int main() {
  constexpr float kEps = 1e-5f;
  constexpr int kRepeat = 10;
  constexpr int kMaxCpuVerifyDim = 1024;
  auto cases = common::LoadOrCreateTestCasesCsv("data/q_path_fusion/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/q_path_fusion_v1_results.csv");
  ofs << "id,rows,cols,gpu_ms,max_abs_diff,check\n";

  for (size_t i = 0; i < cases.size(); ++i) {
    const int rows = cases[i].rows;
    const int cols = cases[i].cols;
    const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const size_t mat = static_cast<size_t>(cols) * static_cast<size_t>(cols);

    std::vector<float> x(n), gamma(cols), wq(mat), bq(cols), q_cpu(n), q_gpu(n);
    common::InitMatrix(x, rows, cols);
    InitVector(gamma);
    common::InitMatrix(wq, cols, cols);
    InitVector(bq);
    const bool do_cpu_verify = (rows <= kMaxCpuVerifyDim && cols <= kMaxCpuVerifyDim);
    if (do_cpu_verify) {
      QPathFusionCPU(x.data(), gamma.data(), wq.data(), bq.data(), q_cpu.data(), rows, cols, kEps);
    }

    // 设备内存
    float *dx = nullptr, *dgamma = nullptr, *dwq = nullptr, *dbq = nullptr, *dq = nullptr;
    CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dgamma, cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dwq, mat * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dbq, cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dq, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dgamma, gamma.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dwq, wq.data(), mat * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dbq, bq.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid(rows);
    dim3 block(QPF_BLOCK_SIZE);
    const size_t smem_size = static_cast<size_t>(cols) * sizeof(float);

    QPathFusionV1Kernel<<<grid, block, smem_size>>>(
        dx, dgamma, dwq, dbq, dq, rows, cols, kEps);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int rep = 0; rep < kRepeat; ++rep) {
      QPathFusionV1Kernel<<<grid, block, smem_size>>>(
          dx, dgamma, dwq, dbq, dq, rows, cols, kEps);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    const float gpu_ms = total_ms / static_cast<float>(kRepeat);

    CHECK_CUDA(cudaMemcpy(q_gpu.data(), dq, n * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = true;
    double max_abs_diff = 0.0;
    const char* check = "SKIP";
    if (do_cpu_verify) {
      ok = common::CheckEqual(q_cpu, q_gpu, 1e-3f);
      max_abs_diff = common::MaxAbsDiff(q_cpu, q_gpu);
      check = ok ? "PASS" : "FAIL";
    }

    std::cout << "rows=" << rows << " cols=" << cols
              << " | " << std::fixed << std::setprecision(4) << gpu_ms << " ms"
              << " | " << check << "\n";
    ofs << cases[i].id << "," << rows << "," << cols << "," << gpu_ms << "," << max_abs_diff
        << "," << check << "\n";
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dgamma));
    CHECK_CUDA(cudaFree(dwq));
    CHECK_CUDA(cudaFree(dbq));
    CHECK_CUDA(cudaFree(dq));
  }

  return 0;
}

