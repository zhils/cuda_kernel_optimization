#ifndef SOFTMAX_KERNELS_CUH_
#define SOFTMAX_KERNELS_CUH_

#include <cuda_runtime.h>

__global__ void SoftmaxNaiveKernel(const float* x, float* y, int rows, int cols) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float maxv = x[r * cols];
  for (int c = 1; c < cols; ++c) maxv = fmaxf(maxv, x[r * cols + c]);
  float sum = 0.f;
  for (int c = 0; c < cols; ++c) {
    float v = __expf(x[r * cols + c] - maxv);
    y[r * cols + c] = v;
    sum += v;
  }
  for (int c = 0; c < cols; ++c) y[r * cols + c] /= sum;
}

constexpr int SOFTMAX_WARP_SIZE = 32;
constexpr int SOFTMAX_WARPS_PER_BLOCK = 4;
constexpr int SOFTMAX_BLOCK_SIZE = SOFTMAX_WARP_SIZE * SOFTMAX_WARPS_PER_BLOCK;

__global__ void SoftmaxV1Kernel(const float* __restrict__ x, float* __restrict__ y, int rows,
                                int cols) {
  const int warp_id = threadIdx.x / SOFTMAX_WARP_SIZE;
  const int lane = threadIdx.x % SOFTMAX_WARP_SIZE;
  const int row = blockIdx.x * SOFTMAX_WARPS_PER_BLOCK + warp_id;
  if (row >= rows) return;

  const float* row_x = x + row * cols;
  float* row_y = y + row * cols;

  extern __shared__ float smem[];
  float* s_exp = smem + warp_id * cols;
  __shared__ float s_max[SOFTMAX_WARPS_PER_BLOCK];
  __shared__ float s_sum[SOFTMAX_WARPS_PER_BLOCK];

  const int cols4 = cols / 4;

  for (int c = lane; c < cols4; c += SOFTMAX_WARP_SIZE) {
    const float4 v = __ldg(reinterpret_cast<const float4*>(row_x + c * 4));
    s_exp[c * 4 + 0] = v.x;
    s_exp[c * 4 + 1] = v.y;
    s_exp[c * 4 + 2] = v.z;
    s_exp[c * 4 + 3] = v.w;
  }
  for (int c = cols4 * 4 + lane; c < cols; c += SOFTMAX_WARP_SIZE) {
    s_exp[c] = __ldg(row_x + c);
  }
  __syncthreads();

  if (lane == 0) {
    float row_max = s_exp[0];
    for (int c = 1; c < cols; ++c) {
      row_max = fmaxf(row_max, s_exp[c]);
    }
    s_max[warp_id] = row_max;

    float row_sum = 0.f;
    for (int c = 0; c < cols; ++c) {
      s_exp[c] = __expf(s_exp[c] - row_max);
      row_sum += s_exp[c];
    }
    s_sum[warp_id] = row_sum;
  }
  __syncthreads();

  const float inv = 1.f / s_sum[warp_id];
  for (int c = lane; c < cols4; c += SOFTMAX_WARP_SIZE) {
    const float4 e = *reinterpret_cast<const float4*>(s_exp + c * 4);
    float4 out;
    out.x = e.x * inv;
    out.y = e.y * inv;
    out.z = e.z * inv;
    out.w = e.w * inv;
    *reinterpret_cast<float4*>(row_y + c * 4) = out;
  }
  for (int c = cols4 * 4 + lane; c < cols; c += SOFTMAX_WARP_SIZE) {
    row_y[c] = s_exp[c] * inv;
  }
}

__global__ void SoftmaxV2Kernel(const float* __restrict__ x, float* __restrict__ y, int rows,
                                int cols) {
  const int warp_id = threadIdx.x / SOFTMAX_WARP_SIZE;
  const int lane = threadIdx.x % SOFTMAX_WARP_SIZE;
  const int row = blockIdx.x * SOFTMAX_WARPS_PER_BLOCK + warp_id;
  if (row >= rows) return;

  const float* row_x = x + row * cols;
  float* row_y = y + row * cols;

  extern __shared__ float smem[];
  float* s_exp = smem + warp_id * cols;
  __shared__ float s_max[SOFTMAX_WARPS_PER_BLOCK];
  __shared__ float s_sum[SOFTMAX_WARPS_PER_BLOCK];

  const int cols4 = cols / 4;

  for (int c = lane; c < cols4; c += SOFTMAX_WARP_SIZE) {
    const float4 v = __ldg(reinterpret_cast<const float4*>(row_x + c * 4));
    s_exp[c * 4 + 0] = v.x;
    s_exp[c * 4 + 1] = v.y;
    s_exp[c * 4 + 2] = v.z;
    s_exp[c * 4 + 3] = v.w;
  }
  for (int c = cols4 * 4 + lane; c < cols; c += SOFTMAX_WARP_SIZE) {
    s_exp[c] = __ldg(row_x + c);
  }
  __syncthreads();

  float local_max = -INFINITY;
  for (int c = lane; c < cols; c += SOFTMAX_WARP_SIZE) {
    local_max = fmaxf(local_max, s_exp[c]);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
  }
  if (lane == 0) s_max[warp_id] = local_max;
  __syncthreads();
  const float row_max = s_max[warp_id];

  float local_sum = 0.f;
  for (int c = lane; c < cols; c += SOFTMAX_WARP_SIZE) {
    const float e = __expf(s_exp[c] - row_max);
    s_exp[c] = e;
    local_sum += e;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  }
  if (lane == 0) s_sum[warp_id] = local_sum;
  __syncthreads();
  const float inv = 1.f / s_sum[warp_id];

  for (int c = lane; c < cols4; c += SOFTMAX_WARP_SIZE) {
    const float4 e = *reinterpret_cast<const float4*>(s_exp + c * 4);
    float4 out;
    out.x = e.x * inv;
    out.y = e.y * inv;
    out.z = e.z * inv;
    out.w = e.w * inv;
    *reinterpret_cast<float4*>(row_y + c * 4) = out;
  }
  for (int c = cols4 * 4 + lane; c < cols; c += SOFTMAX_WARP_SIZE) {
    row_y[c] = s_exp[c] * inv;
  }
}

__global__ void SoftmaxV3Kernel(const float* __restrict__ x, float* __restrict__ y, int rows,
                                int cols) {
  const int warp_id = threadIdx.x / SOFTMAX_WARP_SIZE;
  const int lane = threadIdx.x % SOFTMAX_WARP_SIZE;
  const int row = blockIdx.x * SOFTMAX_WARPS_PER_BLOCK + warp_id;
  if (row >= rows) return;

  const float* row_x = x + row * cols;
  float* row_y = y + row * cols;

  extern __shared__ float smem[];
  float* s_data = smem + warp_id * cols;
  __shared__ float s_max[SOFTMAX_WARPS_PER_BLOCK];
  __shared__ float s_sum[SOFTMAX_WARPS_PER_BLOCK];

  const int cols4 = cols / 4;

#pragma unroll 4
  for (int c = lane; c < cols4; c += SOFTMAX_WARP_SIZE) {
    const float4 v = __ldg(reinterpret_cast<const float4*>(row_x + c * 4));
    s_data[c * 4 + 0] = v.x;
    s_data[c * 4 + 1] = v.y;
    s_data[c * 4 + 2] = v.z;
    s_data[c * 4 + 3] = v.w;
  }
  for (int c = cols4 * 4 + lane; c < cols; c += SOFTMAX_WARP_SIZE) {
    s_data[c] = __ldg(row_x + c);
  }
  __syncthreads();

  float local_max = -INFINITY;
  float local_sum = 0.f;

#pragma unroll 8
  for (int c = lane; c < cols; c += SOFTMAX_WARP_SIZE) {
    const float val = s_data[c];
    if (val > local_max) {
      local_sum *= __expf(local_max - val);
      local_max = val;
    }
    local_sum += __expf(val - local_max);
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float m2 = __shfl_down_sync(0xffffffff, local_max, offset);
    float s2 = __shfl_down_sync(0xffffffff, local_sum, offset);
    float new_max = fmaxf(local_max, m2);
    local_sum = local_sum * __expf(local_max - new_max) + s2 * __expf(m2 - new_max);
    local_max = new_max;
  }
  if (lane == 0) {
    s_max[warp_id] = local_max;
    s_sum[warp_id] = local_sum;
  }
  __syncthreads();

  const float row_max = s_max[warp_id];
  const float row_sum = s_sum[warp_id];
  const float inv = 1.f / row_sum;

#pragma unroll 4
  for (int c = lane; c < cols4; c += SOFTMAX_WARP_SIZE) {
    const float4 d = *reinterpret_cast<const float4*>(s_data + c * 4);
    float4 out;
    out.x = __expf(d.x - row_max) * inv;
    out.y = __expf(d.y - row_max) * inv;
    out.z = __expf(d.z - row_max) * inv;
    out.w = __expf(d.w - row_max) * inv;
    *reinterpret_cast<float4*>(row_y + c * 4) = out;
  }
  for (int c = cols4 * 4 + lane; c < cols; c += SOFTMAX_WARP_SIZE) {
    row_y[c] = __expf(s_data[c] - row_max) * inv;
  }
}

#endif  // SOFTMAX_KERNELS_CUH_
