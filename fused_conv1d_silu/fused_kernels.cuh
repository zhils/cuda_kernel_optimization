#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>

namespace fused_gpu {

constexpr int kBlockSize = 256;
constexpr int kMaxKernelSize = 64;

__global__ void LinearKernel(const float* __restrict__ x, const float* __restrict__ W,
                             const float* __restrict__ b, float* __restrict__ out, int B, int L,
                             int D, int H_out) {
  const int b_idx = blockIdx.x;
  const int t = blockIdx.y;
  const int h = threadIdx.x + blockIdx.z * blockDim.x;
  if (b_idx >= B || t >= L || h >= H_out) return;

  float sum = (b != nullptr) ? b[h] : 0.f;
  const float* x_bt = x + (b_idx * L + t) * D;
  const float* W_h = W + h * D;
  for (int d = 0; d < D; ++d) {
    sum += x_bt[d] * W_h[d];
  }
  out[(b_idx * L + t) * H_out + h] = sum;
}

__global__ void SplitQKVKernel(const float* __restrict__ qkv, float* __restrict__ Q,
                               float* __restrict__ K, float* __restrict__ V, int B, int L,
                               int H) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = B * L * H;
  if (idx >= total) return;

  const int b = idx / (L * H);
  const int rem = idx % (L * H);
  const int t = rem / H;
  const int h = rem % H;

  const int base = ((b * L) + t) * (3 * H);
  Q[idx] = qkv[base + h];
  K[idx] = qkv[base + H + h];
  V[idx] = qkv[base + 2 * H + h];
}

__global__ void CausalConv1dSiLUKernel(const float* __restrict__ u,
                                       const float* __restrict__ K_conv, float* __restrict__ y,
                                       int B, int L, int H, int k_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = B * L * H;
  if (idx >= total) return;

  const int b = idx / (L * H);
  const int rem = idx % (L * H);
  const int t = rem / H;
  const int h = rem % H;

  float sum = 0.f;
  const int k_start = (t < k_size) ? 0 : (t - k_size + 1);
  for (int i = k_start; i <= t; ++i) {
    const int u_idx = ((b * L) + i) * H + h;
    const int k_idx = (t - i) * H + h;
    sum += u[u_idx] * K_conv[k_idx];
  }

  const float sigmoid = 1.f / (1.f + expf(-sum));
  y[idx] = sum * sigmoid;
}

__global__ void GateMulKernel(const float* __restrict__ V_raw, const float* __restrict__ gate,
                              float* __restrict__ V_out, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  V_out[idx] = V_raw[idx] * gate[idx];
}

__global__ void FusedV1Kernel(const float* __restrict__ x, const float* __restrict__ W_qkv,
                              const float* __restrict__ b_qkv, const float* __restrict__ W_z,
                              const float* __restrict__ b_z, const float* __restrict__ K_conv,
                              float* __restrict__ Q, float* __restrict__ K, float* __restrict__ V,
                              int B, int L, int D, int H, int k_size) {
  // 线程映射：每个 block 对应一个 batch，每个线程对应一个通道
  const int b_idx = blockIdx.x;
  const int h = blockIdx.y * blockDim.x + threadIdx.x;
  if (b_idx >= B || h >= H) return;

  // 预取当前通道的权重指针与 bias
  const float* Wq = W_qkv + static_cast<size_t>(h) * D;
  const float* Wk = W_qkv + static_cast<size_t>(h + H) * D;
  const float* Wv = W_qkv + static_cast<size_t>(h + 2 * H) * D;
  const float* Wz = W_z + static_cast<size_t>(h) * D;

  const float bq = b_qkv[h];
  const float bk = b_qkv[h + H];
  const float bv = b_qkv[h + 2 * H];
  const float bz = b_z[h];

  // 环形历史缓存初始化
  float z_hist[kMaxKernelSize];
#pragma unroll
  for (int i = 0; i < kMaxKernelSize; ++i) {
    z_hist[i] = 0.f;
  }

  const int ksize = (k_size <= kMaxKernelSize) ? k_size : kMaxKernelSize;

  // 沿时间步顺序推进
  for (int t = 0; t < L; ++t) {
    const float* x_bt = x + (static_cast<size_t>(b_idx) * L + t) * D;

    float q_raw = bq;
    float k_raw = bk;
    float v_raw = bv;
    float z_proj = bz;

    // q/k/v/z 投影
    for (int d = 0; d < D; ++d) {
      const float xv = x_bt[d];
      q_raw += xv * Wq[d];
      k_raw += xv * Wk[d];
      v_raw += xv * Wv[d];
      z_proj += xv * Wz[d];
    }

    // 将 z 投影写入环形历史缓存
    z_hist[t % ksize] = z_proj;

    // 因果卷积 + SiLU 激活
    float z_conv = 0.f;
    const int valid = (t + 1 < ksize) ? (t + 1) : ksize;
    for (int lag = 0; lag < valid; ++lag) {
      const int hist_idx = (t - lag) % ksize;
      z_conv += z_hist[hist_idx] * K_conv[static_cast<size_t>(lag) * H + h];
    }

    const float sigmoid = 1.f / (1.f + expf(-z_conv));
    const float z_act = z_conv * sigmoid;

    // 回写 Q/K/V 到全局内存
    const size_t out_idx = (static_cast<size_t>(b_idx) * L + t) * H + h;
    Q[out_idx] = q_raw;
    K[out_idx] = k_raw;
    V[out_idx] = v_raw * z_act;
  }
}

// 双 kernel 融合：Kernel A 做 q/k/v/z 全投影（float4 向量化），Kernel B 做因果卷积+门控
__global__ void ComputeQKVZKernel(const float* __restrict__ x, const float* __restrict__ W_qkv,
                                  const float* __restrict__ b_qkv, const float* __restrict__ W_z,
                                  const float* __restrict__ b_z, float* __restrict__ Q,
                                  float* __restrict__ K, float* __restrict__ V,
                                  float* __restrict__ z_proj, int B, int L, int D, int H) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_threads = blockDim.x * gridDim.x;

  // grid-stride loop 遍历所有 (b,t,h)
  for (int idx = global_idx; idx < B * L * H; idx += total_threads) {
    const int b = idx / (L * H);
    const int rem1 = idx % (L * H);
    const int t = rem1 / H;
    const int h = rem1 % H;

    // 预取每通道权重指针
    const float* x_bt = x + (static_cast<size_t>(b) * L + t) * D;
    const float* Wq = W_qkv + static_cast<size_t>(h) * D;
    const float* Wk = W_qkv + static_cast<size_t>(h + H) * D;
    const float* Wv = W_qkv + static_cast<size_t>(h + 2 * H) * D;
    const float* Wz = W_z + static_cast<size_t>(h) * D;

    // 从 bias 初始化累加器
    float q_raw = b_qkv[h];
    float k_raw = b_qkv[h + H];
    float v_raw = b_qkv[h + 2 * H];
    float zp = b_z[h];

    // float4 向量化点积累加
    int d = 0;
    for (; d + 3 < D; d += 4) {
      const float4 xv = reinterpret_cast<const float4*>(x_bt)[d / 4];
      const float4 wq = reinterpret_cast<const float4*>(Wq)[d / 4];
      const float4 wk = reinterpret_cast<const float4*>(Wk)[d / 4];
      const float4 wv = reinterpret_cast<const float4*>(Wv)[d / 4];
      const float4 wz = reinterpret_cast<const float4*>(Wz)[d / 4];
      q_raw += xv.x * wq.x + xv.y * wq.y + xv.z * wq.z + xv.w * wq.w;
      k_raw += xv.x * wk.x + xv.y * wk.y + xv.z * wk.z + xv.w * wk.w;
      v_raw += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
      zp += xv.x * wz.x + xv.y * wz.y + xv.z * wz.z + xv.w * wz.w;
    }

    // 尾部标量累加
    for (; d < D; ++d) {
      const float xv = x_bt[d];
      q_raw += xv * Wq[d];
      k_raw += xv * Wk[d];
      v_raw += xv * Wv[d];
      zp += xv * Wz[d];
    }

    // 回写 Q/K/V 和 z_proj
    const size_t out_idx = (static_cast<size_t>(b) * L + t) * H + h;
    Q[out_idx] = q_raw;
    K[out_idx] = k_raw;
    V[out_idx] = v_raw;
    z_proj[out_idx] = zp;
  }
}

// 因果卷积 + SiLU 门控，就地更新 V
__global__ void ConvGateKernel(const float* __restrict__ z_proj, const float* __restrict__ K_conv,
                               float* __restrict__ V, int B, int L, int H, int k_size) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_threads = blockDim.x * gridDim.x;

  // grid-stride loop 遍历所有 (b,t,h)
  for (int idx = global_idx; idx < B * L * H; idx += total_threads) {
    const int b = idx / (L * H);
    const int rem1 = idx % (L * H);
    const int t = rem1 / H;
    const int h = rem1 % H;

    const size_t base = (static_cast<size_t>(b) * L) * H;

    // 因果卷积
    float z_conv = 0.f;
    for (int i = 0; i < k_size; ++i) {
      const int ti = t - i;
      if (ti < 0) continue;
      z_conv += z_proj[base + static_cast<size_t>(ti) * H + h] *
                K_conv[static_cast<size_t>(i) * H + h];
    }

    // SiLU 激活并就地更新 V
    const float sigmoid = 1.f / (1.f + expf(-z_conv));
    const float z_act = z_conv * sigmoid;

    const size_t out_idx = base + static_cast<size_t>(t) * H + h;
    V[out_idx] *= z_act;
  }
}

// 拆分 QKV 并加 bias：将 CUTLASS GEMM 输出的 (BL, 3H) qkv 拆为 Q/K/V
__global__ void SplitQKVAddBiasKernel(const float* __restrict__ qkv, const float* __restrict__ b_qkv,
                                      float* __restrict__ Q, float* __restrict__ K,
                                      float* __restrict__ V, int BL, int H) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = BL * H;
  if (idx >= total) return;

  const int m = idx / H;
  const int h = idx - m * H;
  const size_t base = static_cast<size_t>(m) * (3 * H);

  Q[idx] = qkv[base + h] + b_qkv[h];
  K[idx] = qkv[base + H + h] + b_qkv[H + h];
  V[idx] = qkv[base + 2 * H + h] + b_qkv[2 * H + h];
}

// 行广播加 bias：对 z 投影张量沿 BL 维广播通道级 bias
__global__ void AddRowBiasKernel(float* __restrict__ z, const float* __restrict__ b, int BL, int H) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BL * H) return;
  z[idx] += b[idx % H];
}

}  // namespace fused_gpu

#include <cuda_bf16.h>

template <typename InT, typename OutT>
struct Convert;

// float → half 转换
template <> struct Convert<float, __half> {
  static __device__ __half f(float v) { return __float2half(v); }
};

// float → bfloat16 转换
template <> struct Convert<float, __nv_bfloat16> {
  static __device__ __nv_bfloat16 f(float v) { return __float2bfloat16(v); }
};

// bfloat16 → float 转换
template <> struct Convert<__nv_bfloat16, float> {
  static __device__ float f(__nv_bfloat16 v) { return __bfloat162float(v); }
};

// int8 → float 转换
template <> struct Convert<int8_t, float> {
  static __device__ float f(int8_t v) { return static_cast<float>(v); }
};

// int32 → float 转换
template <> struct Convert<int32_t, float> {
  static __device__ float f(int32_t v) { return static_cast<float>(v); }
};

// float → int8 量化，含 round + clamp
template <> struct Convert<float, int8_t> {
  static __device__ int8_t f(float v) {
    v = roundf(v);
    v = fminf(fmaxf(v, -128.0f), 127.0f);
    return static_cast<int8_t>(v);
  }
};

// 通用类型转换 kernel，逐元素调用 Convert<InT, OutT>::f
template <typename InT, typename OutT>
__global__ void ConvertKernel(const InT* __restrict__ in, OutT* __restrict__ out, int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  out[idx] = Convert<InT, OutT>::f(in[idx]);
}

// 转置转换 kernel：将 float (N,K) 行主序转为 OutT (K,N) 列主序
template <typename OutT>
__global__ void ConvertTransposeKernel(
    const float* __restrict__ in, int N, int K, int ld_in,
    OutT* __restrict__ out, int ld_out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N * K;
  if (idx >= total) return;
  const int n = idx / K;
  const int k = idx % K;
  out[k * ld_out + n] = Convert<float, OutT>::f(in[n * ld_in + k]);
}

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
#include <cuda_fp8.h>

// fp8 e4m3 → float 转换
template <> struct Convert<__nv_fp8_e4m3, float> {
  static __device__ float f(__nv_fp8_e4m3 v) { return static_cast<float>(v); }
};

// float → fp8 e4m3 转换
template <> struct Convert<float, __nv_fp8_e4m3> {
  static __device__ __nv_fp8_e4m3 f(float v) { return static_cast<__nv_fp8_e4m3>(v); }
};

// fp8 e5m2 → float 转换
template <> struct Convert<__nv_fp8_e5m2, float> {
  static __device__ float f(__nv_fp8_e5m2 v) { return static_cast<float>(v); }
};

// float → fp8 e5m2 转换
template <> struct Convert<float, __nv_fp8_e5m2> {
  static __device__ __nv_fp8_e5m2 f(float v) { return static_cast<__nv_fp8_e5m2>(v); }
};
#endif
