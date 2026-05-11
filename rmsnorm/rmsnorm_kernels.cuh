#ifndef RMSNORM_KERNELS_CUH_
#define RMSNORM_KERNELS_CUH_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

__global__ void RMSNormV0Kernel(
  const float* x, 
  float* y, 
  const float* weight, 
  int rows, 
  int cols,
  float eps
) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float sq_sum = 0.f;
  for (int c = 0; c < cols; ++c) {
    float val = x[r * cols + c];
    sq_sum += val * val;
  }
  float rms = rsqrtf(sq_sum / cols + eps);
  for (int c = 0; c < cols; ++c) {
    y[r * cols + c] = x[r * cols + c] * rms * weight[c];
  }
}

constexpr int RMSNORM_WARP_SIZE = 32;
constexpr int RMSNORM_WARPS_PER_BLOCK = 4;
constexpr int RMSNORM_BLOCK_SIZE = RMSNORM_WARP_SIZE * RMSNORM_WARPS_PER_BLOCK;

__global__ void RMSNormV1Kernel(
  const float* __restrict__ x, 
  float* __restrict__ y,
  const float* __restrict__ weight, 
  int rows, 
  int cols, 
  float eps
) {
  const int warp_id = threadIdx.x / RMSNORM_WARP_SIZE;
  const int lane = threadIdx.x % RMSNORM_WARP_SIZE;
  const int row = blockIdx.x * RMSNORM_WARPS_PER_BLOCK + warp_id;
  if (row >= rows) return;

  const float* row_x = x + row * cols;
  float* row_y = y + row * cols;

  __shared__ float s_sq_sum[RMSNORM_WARPS_PER_BLOCK];
  extern __shared__ float s_data[];
  float* s_row = s_data + warp_id * cols;

  const int cols4 = cols / 4;

  for (int c = lane; c < cols4; c += RMSNORM_WARP_SIZE) {
    const float4 v = *reinterpret_cast<const float4*>(row_x + c * 4);
    s_row[c * 4 + 0] = v.x;
    s_row[c * 4 + 1] = v.y;
    s_row[c * 4 + 2] = v.z;
    s_row[c * 4 + 3] = v.w;
  }
  for (int c = cols4 * 4 + lane; c < cols; c += RMSNORM_WARP_SIZE) {
    s_row[c] = row_x[c];
  }
  __syncthreads();

  if (lane == 0) {
    float sq_sum = 0.f;
    for (int c = 0; c < cols; ++c) {
      const float val = s_row[c];
      sq_sum += val * val;
    }
    s_sq_sum[warp_id] = sq_sum;
  }
  __syncthreads();

  const float rms = rsqrtf(s_sq_sum[warp_id] / static_cast<float>(cols) + eps);
  for (int c = lane; c < cols4; c += RMSNORM_WARP_SIZE) {
    const float4 vx = *reinterpret_cast<const float4*>(s_row + c * 4);
    const float4 vw = *reinterpret_cast<const float4*>(weight + c * 4);
    float4 vy;
    vy.x = vx.x * rms * vw.x;
    vy.y = vx.y * rms * vw.y;
    vy.z = vx.z * rms * vw.z;
    vy.w = vx.w * rms * vw.w;
    *reinterpret_cast<float4*>(row_y + c * 4) = vy;
  }
  for (int c = cols4 * 4 + lane; c < cols; c += RMSNORM_WARP_SIZE) {
    row_y[c] = s_row[c] * rms * weight[c];
  }
}

__global__ void RMSNormV2Kernel(
  const float* __restrict__ x, 
  float* __restrict__ y,
  const float* __restrict__ weight, 
  int rows, 
  int cols, 
  float eps
) {
  const int warp_id = threadIdx.x / RMSNORM_WARP_SIZE;
  const int lane = threadIdx.x % RMSNORM_WARP_SIZE;
  const int row = blockIdx.x * RMSNORM_WARPS_PER_BLOCK + warp_id;
  if (row >= rows) return;

  const float* row_x = x + row * cols;
  float* row_y = y + row * cols;

  __shared__ float s_sq_sum[RMSNORM_WARPS_PER_BLOCK];
  extern __shared__ float s_data[];
  float* s_row = s_data + warp_id * cols;

  const int cols4 = cols / 4;
  for (int c = lane; c < cols4; c += RMSNORM_WARP_SIZE) {
    const float4 v = *reinterpret_cast<const float4*>(row_x + c * 4);
    s_row[c * 4 + 0] = v.x;
    s_row[c * 4 + 1] = v.y;
    s_row[c * 4 + 2] = v.z;
    s_row[c * 4 + 3] = v.w;
  }
  for (int c = cols4 * 4 + lane; c < cols; c += RMSNORM_WARP_SIZE) {
    s_row[c] = row_x[c];
  }
  __syncthreads();

  float local_sum = 0.f;
  for (int c = lane; c < cols; c += RMSNORM_WARP_SIZE) {
    const float val = s_row[c];
    local_sum += val * val;
  }
  // 区别只是变成了规约求和
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  }
  if (lane == 0) {
    s_sq_sum[warp_id] = local_sum;
  }
  __syncthreads();

  const float rms = rsqrtf(s_sq_sum[warp_id] / static_cast<float>(cols) + eps);
  for (int c = lane; c < cols4; c += RMSNORM_WARP_SIZE) {
    const float4 vx = *reinterpret_cast<const float4*>(s_row + c * 4);
    const float4 vw = *reinterpret_cast<const float4*>(weight + c * 4);
    float4 vy;
    vy.x = vx.x * rms * vw.x;
    vy.y = vx.y * rms * vw.y;
    vy.z = vx.z * rms * vw.z;
    vy.w = vx.w * rms * vw.w;
    *reinterpret_cast<float4*>(row_y + c * 4) = vy;
  }
  for (int c = cols4 * 4 + lane; c < cols; c += RMSNORM_WARP_SIZE) {
    row_y[c] = s_row[c] * rms * weight[c];
  }
}

__global__ void RMSNormV3Kernel(
  const float* __restrict__ x, 
  float* __restrict__ y,
  const float* __restrict__ weight, 
  int rows, 
  int cols, 
  float eps
) {
  const int warp_id = threadIdx.x / RMSNORM_WARP_SIZE;
  const int lane = threadIdx.x % RMSNORM_WARP_SIZE;
  const int row = blockIdx.x * RMSNORM_WARPS_PER_BLOCK + warp_id;

  extern __shared__ float s_weight[];
  for (int c = threadIdx.x; c < cols; c += RMSNORM_BLOCK_SIZE) {
    s_weight[c] = weight[c];
  }
  __syncthreads();

  if (row >= rows) return;

  const float* row_x = x + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  float* row_y = y + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);

  const int cols4 = cols / 4;
  const bool align4 = (cols % 4 == 0) &&
                      (reinterpret_cast<std::uintptr_t>(row_x) % 16u == 0u) &&
                      (reinterpret_cast<std::uintptr_t>(row_y) % 16u == 0u) &&
                      (reinterpret_cast<std::uintptr_t>(weight) % 16u == 0u);

  float local_sum = 0.f;
  if (align4) {
    for (int c = lane; c < cols4; c += RMSNORM_WARP_SIZE) {
      const float4 v = __ldg(reinterpret_cast<const float4*>(row_x + c * 4));
      local_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
  } else {
    for (int c = lane; c < cols; c += RMSNORM_WARP_SIZE) {
      const float val = __ldg(row_x + c);
      local_sum += val * val;
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  }
  const float sq_sum = __shfl_sync(0xffffffff, local_sum, 0);
  const float rms = rsqrtf(sq_sum / static_cast<float>(cols) + eps);

  if (align4) {
    for (int c = lane; c < cols4; c += RMSNORM_WARP_SIZE) {
      const float4 vx = __ldg(reinterpret_cast<const float4*>(row_x + c * 4));
      const float4 vw = *reinterpret_cast<const float4*>(s_weight + c * 4);
      *reinterpret_cast<float4*>(row_y + c * 4) =
          make_float4(vx.x * rms * vw.x, vx.y * rms * vw.y, vx.z * rms * vw.z, vx.w * rms * vw.w);
    }
  } else {
    for (int c = lane; c < cols; c += RMSNORM_WARP_SIZE) {
      row_y[c] = __ldg(row_x + c) * rms * s_weight[c];
    }
  }
}

#endif  // RMSNORM_KERNELS_CUH_
