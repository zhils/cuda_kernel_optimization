#pragma once

#include <cuda_runtime.h>

#ifndef SOFTMAX_WARP_SIZE
#define SOFTMAX_WARP_SIZE 32
#endif

#ifndef SOFTMAX_WARPS_PER_BLOCK
#define SOFTMAX_WARPS_PER_BLOCK 4
#endif

#ifndef SOFTMAX_BLOCK_SIZE
#define SOFTMAX_BLOCK_SIZE (SOFTMAX_WARP_SIZE * SOFTMAX_WARPS_PER_BLOCK)
#endif

__global__ void SoftmaxNaiveKernel(const float* x, float* y, int rows, int cols);

__global__ void SoftmaxV1Kernel(const float* __restrict__ x, float* __restrict__ y,
                                int rows, int cols);

__global__ void SoftmaxV2Kernel(const float* __restrict__ x, float* __restrict__ y,
                                int rows, int cols);

__global__ void SoftmaxV3Kernel(const float* __restrict__ x, float* __restrict__ y,
                                int rows, int cols);
