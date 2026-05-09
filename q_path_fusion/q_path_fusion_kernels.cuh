#pragma once

#include <cuda_runtime.h>

constexpr int QPF_BLOCK_SIZE = 256;
constexpr int QPF_WARP_SIZE = 32;

__global__ void QPathFusionV0Kernel(
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
  const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(cols);
  const float* x_row = x + row_offset;

  extern __shared__ float s_data[];
  float* s_sq = s_data;

  float local_sq = 0.0f;
  for (int c = tx; c < cols; c += blockDim.x) {
    const float v = x_row[c];
    local_sq += v * v;
  }
  s_sq[tx] = local_sq;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tx < stride) {
      s_sq[tx] += s_sq[tx + stride];
    }
    __syncthreads();
  }

  const float inv_rms = rsqrtf(s_sq[0] / static_cast<float>(cols) + eps);

  for (int out_col = tx; out_col < cols; out_col += blockDim.x) {
    float acc = bq[out_col];
    for (int k = 0; k < cols; ++k) {
      const float norm_val = x_row[k] * inv_rms * gamma[k];
      acc += norm_val * wq[static_cast<size_t>(k) * static_cast<size_t>(cols) + out_col];
    }
    q[row_offset + out_col] = acc;
  }
}

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

  const int lane = threadIdx.x % QPF_WARP_SIZE;
  const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(cols);
  const float* x_row = x + row_offset;

  float local_sq = 0.0f;
  for (int c = lane; c < cols; c += QPF_WARP_SIZE) {
    const float v = x_row[c];
    local_sq += v * v;
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sq += __shfl_down_sync(0xffffffff, local_sq, offset);
  }
  const float sq_sum = __shfl_sync(0xffffffff, local_sq, 0);
  const float inv_rms = rsqrtf(sq_sum / static_cast<float>(cols) + eps);

  for (int out_col = lane; out_col < cols; out_col += QPF_WARP_SIZE) {
    float acc = bq[out_col];
    for (int k = 0; k < cols; ++k) {
      const float norm_val = x_row[k] * inv_rms * gamma[k];
      acc += norm_val * wq[static_cast<size_t>(k) * static_cast<size_t>(cols) + out_col];
    }
    q[row_offset + out_col] = acc;
  }
}