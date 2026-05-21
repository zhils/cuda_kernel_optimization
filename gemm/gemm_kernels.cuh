#pragma once

#include <cuda_runtime.h>

namespace gemm_gpu {

// 朴素GEMM核函数：每个线程计算输出矩阵的一个元素
__global__ void GemmNaiveKernel(const float* A, const float* B, float* C, int M, int N, int K) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float sum = 0.f;
    for (int k = 0; k < K; ++k) {
      sum += A[static_cast<size_t>(row) * K + k] * B[static_cast<size_t>(k) * N + col];
    }
    C[static_cast<size_t>(row) * N + col] = sum;
  }
}

}  // namespace gemm_gpu
