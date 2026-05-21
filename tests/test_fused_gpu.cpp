#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "common/benchmark.h"
#include "common/test_case.h"
#include "fused/fused_api.h"
#include "fused/fused_ref_cpu.h"

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#endif

#ifndef CKO_SOURCE_DIR
#define CKO_SOURCE_DIR "."
#endif

namespace {

void RequireCudaDevice() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    GTEST_SKIP() << "No CUDA device available";
  }
}

#define TEST_CUDA(call) ASSERT_EQ((call), cudaSuccess)

void FillRandom(std::vector<float>& x, int seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  for (float& v : x) v = dist(gen);
}

struct HostFusedBuffers {
  std::vector<float> x;
  std::vector<float> W_qkv, b_qkv;
  std::vector<float> W_z, b_z;
  std::vector<float> K_conv;
  std::vector<float> Q_ref, K_ref, V_ref;
};

struct DeviceFusedBuffers {
  float *x = nullptr, *W_qkv = nullptr, *b_qkv = nullptr;
  float *W_z = nullptr, *b_z = nullptr, *K_conv = nullptr;
  float *Q = nullptr, *K = nullptr, *V = nullptr;

  ~DeviceFusedBuffers() {
    if (x) cudaFree(x);
    if (W_qkv) cudaFree(W_qkv);
    if (b_qkv) cudaFree(b_qkv);
    if (W_z) cudaFree(W_z);
    if (b_z) cudaFree(b_z);
    if (K_conv) cudaFree(K_conv);
    if (Q) cudaFree(Q);
    if (K) cudaFree(K);
    if (V) cudaFree(V);
  }
};

void PrepareHostCase(HostFusedBuffers& h, int B, int L, int D, int H, int k_size) {
  h.x.assign(static_cast<size_t>(B) * L * D, 0.f);
  h.W_qkv.assign(static_cast<size_t>(3) * H * D, 0.f);
  h.b_qkv.assign(static_cast<size_t>(3) * H, 0.f);
  h.W_z.assign(static_cast<size_t>(H) * D, 0.f);
  h.b_z.assign(H, 0.f);
  h.K_conv.assign(static_cast<size_t>(k_size) * H, 0.f);
  h.Q_ref.assign(static_cast<size_t>(B) * L * H, 0.f);
  h.K_ref.assign(static_cast<size_t>(B) * L * H, 0.f);
  h.V_ref.assign(static_cast<size_t>(B) * L * H, 0.f);

  FillRandom(h.x, 42);
  FillRandom(h.W_qkv, 43);
  FillRandom(h.b_qkv, 44);
  FillRandom(h.W_z, 45);
  FillRandom(h.b_z, 46);
  FillRandom(h.K_conv, 47);

  fused::FusedParams ref;
  ref.B = B;
  ref.L = L;
  ref.D = D;
  ref.H = H;
  ref.k_size = k_size;
  ref.dtype = common::DType::kFp32;
  ref.x = h.x.data();
  ref.W_qkv = h.W_qkv.data();
  ref.b_qkv = h.b_qkv.data();
  ref.W_z = h.W_z.data();
  ref.b_z = h.b_z.data();
  ref.K_conv = h.K_conv.data();
  ref.Q = h.Q_ref.data();
  ref.K = h.K_ref.data();
  ref.V = h.V_ref.data();
  ASSERT_TRUE(fused::FusedReferenceHost(ref).ok());
}

void UploadCase(const HostFusedBuffers& h, DeviceFusedBuffers& d) {
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.x), h.x.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.W_qkv), h.W_qkv.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.b_qkv), h.b_qkv.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.W_z), h.W_z.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.b_z), h.b_z.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.K_conv), h.K_conv.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.Q), h.Q_ref.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.K), h.K_ref.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.V), h.V_ref.size() * sizeof(float)));

  TEST_CUDA(cudaMemcpy(d.x, h.x.data(), h.x.size() * sizeof(float), cudaMemcpyHostToDevice));
  TEST_CUDA(
      cudaMemcpy(d.W_qkv, h.W_qkv.data(), h.W_qkv.size() * sizeof(float), cudaMemcpyHostToDevice));
  TEST_CUDA(
      cudaMemcpy(d.b_qkv, h.b_qkv.data(), h.b_qkv.size() * sizeof(float), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(d.W_z, h.W_z.data(), h.W_z.size() * sizeof(float), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(d.b_z, h.b_z.data(), h.b_z.size() * sizeof(float), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(d.K_conv, h.K_conv.data(), h.K_conv.size() * sizeof(float),
                      cudaMemcpyHostToDevice));
}

void RunGpuCase(int B, int L, int D, int H, int k_size, fused::ImplId impl,
                common::DType dtype = common::DType::kFp32) {
  HostFusedBuffers host;
  DeviceFusedBuffers dev;
  PrepareHostCase(host, B, L, D, H, k_size);
  UploadCase(host, dev);

  fused::FusedParams p;
  p.B = B;
  p.L = L;
  p.D = D;
  p.H = H;
  p.k_size = k_size;
  p.dtype = common::DType::kFp32;
  p.x = dev.x;
  p.W_qkv = dev.W_qkv;
  p.b_qkv = dev.b_qkv;
  p.W_z = dev.W_z;
  p.b_z = dev.b_z;
  p.K_conv = dev.K_conv;
  p.Q = dev.Q;
  p.K = dev.K;
  p.V = dev.V;
  p.impl = impl;
  p.dtype = dtype;

  ASSERT_TRUE(fused::FusedRun(p, nullptr).ok());
  TEST_CUDA(cudaDeviceSynchronize());

  std::vector<float> Q_gpu(host.Q_ref.size());
  std::vector<float> K_gpu(host.K_ref.size());
  std::vector<float> V_gpu(host.V_ref.size());
  TEST_CUDA(cudaMemcpy(Q_gpu.data(), dev.Q, Q_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
  TEST_CUDA(cudaMemcpy(K_gpu.data(), dev.K, K_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
  TEST_CUDA(cudaMemcpy(V_gpu.data(), dev.V, V_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

  const float tol = fused::DefaultAbsTolerance();
  EXPECT_TRUE(common::CheckEqual(host.Q_ref, Q_gpu, tol))
      << "max_q=" << common::MaxAbsDiff(host.Q_ref, Q_gpu);
  EXPECT_TRUE(common::CheckEqual(host.K_ref, K_gpu, tol))
      << "max_k=" << common::MaxAbsDiff(host.K_ref, K_gpu);
  EXPECT_TRUE(common::CheckEqual(host.V_ref, V_gpu, tol))
      << "max_v=" << common::MaxAbsDiff(host.V_ref, V_gpu);
}

}  // namespace

TEST(FusedGpuTest, V3MatchesV2Outputs) {
  RequireCudaDevice();
  HostFusedBuffers host;
  DeviceFusedBuffers dev;
  PrepareHostCase(host, 1, 64, 32, 16, 4);
  UploadCase(host, dev);

  fused::FusedParams p2;
  p2.B = 1;
  p2.L = 64;
  p2.D = 32;
  p2.H = 16;
  p2.k_size = 4;
  p2.dtype = common::DType::kFp32;
  p2.x = dev.x;
  p2.W_qkv = dev.W_qkv;
  p2.b_qkv = dev.b_qkv;
  p2.W_z = dev.W_z;
  p2.b_z = dev.b_z;
  p2.K_conv = dev.K_conv;
  p2.Q = dev.Q;
  p2.K = dev.K;
  p2.V = dev.V;
  p2.impl = fused::ImplId::kV2;
  ASSERT_TRUE(fused::FusedRun(p2, nullptr).ok());

  std::vector<float> V_v2(1 * 64 * 16);
  TEST_CUDA(cudaMemcpy(V_v2.data(), dev.V, V_v2.size() * sizeof(float), cudaMemcpyDeviceToHost));

  fused::FusedParams p3 = p2;
  p3.impl = fused::ImplId::kV3;
  ASSERT_TRUE(fused::FusedRun(p3, nullptr).ok());

  std::vector<float> V_v3(V_v2.size());
  TEST_CUDA(cudaMemcpy(V_v3.data(), dev.V, V_v3.size() * sizeof(float), cudaMemcpyDeviceToHost));

  EXPECT_LT(common::MaxAbsDiff(V_v2, V_v3), 2e-2);
}

TEST(FusedGpuTest, SmokeV3) {
  RequireCudaDevice();
  RunGpuCase(1, 128, 64, 32, 4, fused::ImplId::kV3);
}

TEST(FusedGpuTest, LegacyV2) {
  RequireCudaDevice();
  RunGpuCase(1, 64, 32, 16, 4, fused::ImplId::kV2);
}

TEST(FusedGpuTest, LegacyV0) {
  RequireCudaDevice();
  RunGpuCase(1, 32, 16, 8, 2, fused::ImplId::kV0);
}

TEST(FusedGpuTest, CatalogPassCases) {
  RequireCudaDevice();
  const auto cat = common::LoadTestCaseCatalog(
      std::string(CKO_SOURCE_DIR) + "/configs/test_cases/fused_conv1d_silu.json");
  for (const auto& c : cat.cases) {
    if (c.expect != common::ExpectStatus::kPass) continue;
    RunGpuCase(c.fused.B, c.fused.L, c.fused.D, c.fused.H, c.fused.k_size, fused::ImplId::kAuto, common::DType::kFp32);
  }
}

void UploadCaseSmoke(DeviceFusedBuffers& d,
                     const std::vector<float>& x, const std::vector<float>& Wq,
                     const std::vector<float>& bq, const std::vector<float>& Wz,
                     const std::vector<float>& bz, const std::vector<float>& Kc,
                     std::vector<__nv_bfloat16>& Q, std::vector<__nv_bfloat16>& K,
                     std::vector<__nv_bfloat16>& V);

void UploadCaseSmoke(DeviceFusedBuffers& d,
                     const std::vector<float>& x, const std::vector<float>& Wq,
                     const std::vector<float>& bq, const std::vector<float>& Wz,
                     const std::vector<float>& bz, const std::vector<float>& Kc,
                     std::vector<__half>& Q, std::vector<__half>& K,
                     std::vector<__half>& V);

TEST(FusedGpuTest, Fp16Smoke) {
  RequireCudaDevice();

  fused::FusedParams p;
  p.B = 1; p.L = 64; p.D = 32; p.H = 16; p.k_size = 4;
  p.impl = fused::ImplId::kAuto;
  p.dtype = common::DType::kFp16;

  std::vector<float> x_fp32(static_cast<size_t>(p.B) * p.L * p.D, 0.1f);
  std::vector<float> Wq_fp32(static_cast<size_t>(p.D) * p.H * 3, 0.01f);
  std::vector<float> bq_fp32(static_cast<size_t>(p.H) * 3, 0.0f);
  std::vector<float> Wz_fp32(static_cast<size_t>(p.D) * p.H, 0.01f);
  std::vector<float> bz_fp32(p.H, 0.0f);
  std::vector<float> Kc_fp32(static_cast<size_t>(p.H) * p.k_size, 0.01f);

  std::vector<__half> x_f16(x_fp32.size()), Q_f16(static_cast<size_t>(p.B) * p.L * p.H),
      K_f16(Q_f16.size()), V_f16(Q_f16.size());
  for (size_t i = 0; i < x_fp32.size(); ++i) x_f16[i] = __float2half(x_fp32[i]);

  DeviceFusedBuffers dev;
  UploadCaseSmoke(dev, x_fp32, Wq_fp32, bq_fp32, Wz_fp32, bz_fp32, Kc_fp32, Q_f16, K_f16, V_f16);

  __half* d_x = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_x), x_f16.size() * sizeof(__half));
  cudaMemcpy(d_x, x_f16.data(), x_f16.size() * sizeof(__half), cudaMemcpyHostToDevice);

  p.x = d_x;
  p.W_qkv = dev.W_qkv; p.b_qkv = dev.b_qkv;
  p.W_z = dev.W_z; p.b_z = dev.b_z;
  p.K_conv = dev.K_conv;
  p.Q = dev.Q; p.K = dev.K; p.V = dev.V;
  {
    auto st = fused::FusedRun(p, nullptr);
    ASSERT_TRUE(st.ok());
  }
  cudaDeviceSynchronize();
  cudaFree(d_x);
}

void UploadCaseSmoke(DeviceFusedBuffers& d,
                     const std::vector<float>& x, const std::vector<float>& Wq,
                     const std::vector<float>& bq, const std::vector<float>& Wz,
                     const std::vector<float>& bz, const std::vector<float>& Kc,
                     std::vector<__half>& Q, std::vector<__half>& K,
                     std::vector<__half>& V) {
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.W_qkv), Wq.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.b_qkv), bq.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.W_z), Wz.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.b_z), bz.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.K_conv), Kc.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.Q), Q.size() * sizeof(__half)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.K), K.size() * sizeof(__half)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.V), V.size() * sizeof(__half)));
  TEST_CUDA(cudaMemcpy(d.W_qkv, Wq.data(), Wq.size() * sizeof(float), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(d.b_qkv, bq.data(), bq.size() * sizeof(float), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(d.W_z, Wz.data(), Wz.size() * sizeof(float), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(d.b_z, bz.data(), bz.size() * sizeof(float), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(d.K_conv, Kc.data(), Kc.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void UploadCaseSmoke(DeviceFusedBuffers& d,
                     const std::vector<float>& x, const std::vector<float>& Wq,
                     const std::vector<float>& bq, const std::vector<float>& Wz,
                     const std::vector<float>& bz, const std::vector<float>& Kc,
                     std::vector<__nv_bfloat16>& Q, std::vector<__nv_bfloat16>& K,
                     std::vector<__nv_bfloat16>& V) {
  // Q/K/V are bf16 output buffers, weights are fp32
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.W_qkv), Wq.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.b_qkv), bq.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.W_z), Wz.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.b_z), bz.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.K_conv), Kc.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.Q), Q.size() * sizeof(__nv_bfloat16)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.K), K.size() * sizeof(__nv_bfloat16)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.V), V.size() * sizeof(__nv_bfloat16)));
  TEST_CUDA(cudaMemcpy(d.W_qkv, Wq.data(), Wq.size() * sizeof(float), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(d.b_qkv, bq.data(), bq.size() * sizeof(float), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(d.W_z, Wz.data(), Wz.size() * sizeof(float), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(d.b_z, bz.data(), bz.size() * sizeof(float), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(d.K_conv, Kc.data(), Kc.size() * sizeof(float), cudaMemcpyHostToDevice));
}

TEST(FusedGpuTest, Int8Smoke) {
  RequireCudaDevice();
  constexpr int B = 1, L = 64, D = 32, H = 16, k_size = 4;

  HostFusedBuffers host;
  PrepareHostCase(host, B, L, D, H, k_size);

  std::vector<int8_t> x_i8(host.x.size());
  for (size_t i = 0; i < host.x.size(); ++i) {
    float v = roundf(host.x[i]);
    v = fminf(fmaxf(v, -128.0f), 127.0f);
    x_i8[i] = static_cast<int8_t>(v);
  }

  DeviceFusedBuffers dev;
  UploadCase(host, dev);

  int8_t* d_x = nullptr;
  int8_t *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr;
  size_t n_out = static_cast<size_t>(B) * L * H;
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_x), x_i8.size() * sizeof(int8_t)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_Q), n_out * sizeof(int8_t)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_K), n_out * sizeof(int8_t)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_V), n_out * sizeof(int8_t)));
  TEST_CUDA(cudaMemcpy(d_x, x_i8.data(), x_i8.size() * sizeof(int8_t), cudaMemcpyHostToDevice));

  fused::FusedParams p;
  p.B = B; p.L = L; p.D = D; p.H = H; p.k_size = k_size;
  p.dtype = common::DType::kInt8;
  p.x = d_x;
  p.W_qkv = dev.W_qkv; p.b_qkv = dev.b_qkv;
  p.W_z = dev.W_z; p.b_z = dev.b_z;
  p.K_conv = dev.K_conv;
  p.Q = d_Q; p.K = d_K; p.V = d_V;
  p.impl = fused::ImplId::kAuto;

  auto st = fused::FusedRun(p, nullptr);
  if (!st.ok()) std::cerr << "FP8E4M3 FAIL: " << st.message << "\n";
  ASSERT_TRUE(st.ok());
  TEST_CUDA(cudaDeviceSynchronize());

  std::vector<int8_t> Q_i8(n_out), K_i8(n_out), V_i8(n_out);
  TEST_CUDA(cudaMemcpy(Q_i8.data(), d_Q, n_out * sizeof(int8_t), cudaMemcpyDeviceToHost));
  TEST_CUDA(cudaMemcpy(K_i8.data(), d_K, n_out * sizeof(int8_t), cudaMemcpyDeviceToHost));
  TEST_CUDA(cudaMemcpy(V_i8.data(), d_V, n_out * sizeof(int8_t), cudaMemcpyDeviceToHost));

  std::vector<float> Q_fp32(n_out), K_fp32(n_out), V_fp32(n_out);
  for (size_t i = 0; i < n_out; ++i) {
    Q_fp32[i] = static_cast<float>(Q_i8[i]);
    K_fp32[i] = static_cast<float>(K_i8[i]);
    V_fp32[i] = static_cast<float>(V_i8[i]);
  }

  float tol = 2.0f;
  EXPECT_TRUE(common::CheckEqual(host.Q_ref, Q_fp32, tol))
      << "max_q=" << common::MaxAbsDiff(host.Q_ref, Q_fp32);
  EXPECT_TRUE(common::CheckEqual(host.K_ref, K_fp32, tol))
      << "max_k=" << common::MaxAbsDiff(host.K_ref, K_fp32);
  EXPECT_TRUE(common::CheckEqual(host.V_ref, V_fp32, tol))
      << "max_v=" << common::MaxAbsDiff(host.V_ref, V_fp32);

  cudaFree(d_x); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
}

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
TEST(FusedGpuTest, Fp8E4M3Smoke) {
  RequireCudaDevice();
  constexpr int B = 1, L = 64, D = 32, H = 16, k_size = 4;

  HostFusedBuffers host;
  PrepareHostCase(host, B, L, D, H, k_size);

  std::vector<__nv_fp8_e4m3> x_f8(host.x.size());
  for (size_t i = 0; i < host.x.size(); ++i) {
    x_f8[i] = static_cast<__nv_fp8_e4m3>(host.x[i]);
  }

  DeviceFusedBuffers dev;
  UploadCase(host, dev);

  __nv_fp8_e4m3 *d_x = nullptr, *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr;
  size_t n_out = static_cast<size_t>(B) * L * H;
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_x), x_f8.size()));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_Q), n_out));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_K), n_out));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_V), n_out));
  TEST_CUDA(cudaMemcpy(d_x, x_f8.data(), x_f8.size(), cudaMemcpyHostToDevice));

  fused::FusedParams p;
  p.B = B; p.L = L; p.D = D; p.H = H; p.k_size = k_size;
  p.dtype = common::DType::kFp8E4M3;
  p.x = d_x;
  p.W_qkv = dev.W_qkv; p.b_qkv = dev.b_qkv;
  p.W_z = dev.W_z; p.b_z = dev.b_z;
  p.K_conv = dev.K_conv;
  p.Q = d_Q; p.K = d_K; p.V = d_V;
  p.impl = fused::ImplId::kAuto;

  ASSERT_TRUE(fused::FusedRun(p, nullptr).ok());
  TEST_CUDA(cudaDeviceSynchronize());

  std::vector<__nv_fp8_e4m3> Q_f8(n_out), K_f8(n_out), V_f8(n_out);
  TEST_CUDA(cudaMemcpy(Q_f8.data(), d_Q, n_out, cudaMemcpyDeviceToHost));
  TEST_CUDA(cudaMemcpy(K_f8.data(), d_K, n_out, cudaMemcpyDeviceToHost));
  TEST_CUDA(cudaMemcpy(V_f8.data(), d_V, n_out, cudaMemcpyDeviceToHost));

  std::vector<float> Q_fp32(n_out), K_fp32(n_out), V_fp32(n_out);
  for (size_t i = 0; i < n_out; ++i) {
    Q_fp32[i] = static_cast<float>(Q_f8[i]);
    K_fp32[i] = static_cast<float>(K_f8[i]);
    V_fp32[i] = static_cast<float>(V_f8[i]);
  }

  float tol = 0.5f;
  EXPECT_TRUE(common::CheckEqual(host.Q_ref, Q_fp32, tol))
      << "max_q=" << common::MaxAbsDiff(host.Q_ref, Q_fp32);
  EXPECT_TRUE(common::CheckEqual(host.K_ref, K_fp32, tol))
      << "max_k=" << common::MaxAbsDiff(host.K_ref, K_fp32);
  EXPECT_TRUE(common::CheckEqual(host.V_ref, V_fp32, tol))
      << "max_v=" << common::MaxAbsDiff(host.V_ref, V_fp32);

  cudaFree(d_x); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
}

TEST(FusedGpuTest, Fp8E5M2Smoke) {
  RequireCudaDevice();
  constexpr int B = 1, L = 64, D = 32, H = 16, k_size = 4;

  HostFusedBuffers host;
  PrepareHostCase(host, B, L, D, H, k_size);

  std::vector<__nv_fp8_e5m2> x_f8(host.x.size());
  for (size_t i = 0; i < host.x.size(); ++i) {
    x_f8[i] = static_cast<__nv_fp8_e5m2>(host.x[i]);
  }

  DeviceFusedBuffers dev;
  UploadCase(host, dev);

  __nv_fp8_e5m2 *d_x = nullptr, *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr;
  size_t n_out = static_cast<size_t>(B) * L * H;
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_x), x_f8.size()));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_Q), n_out));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_K), n_out));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_V), n_out));
  TEST_CUDA(cudaMemcpy(d_x, x_f8.data(), x_f8.size(), cudaMemcpyHostToDevice));

  fused::FusedParams p;
  p.B = B; p.L = L; p.D = D; p.H = H; p.k_size = k_size;
  p.dtype = common::DType::kFp8E5M2;
  p.x = d_x;
  p.W_qkv = dev.W_qkv; p.b_qkv = dev.b_qkv;
  p.W_z = dev.W_z; p.b_z = dev.b_z;
  p.K_conv = dev.K_conv;
  p.Q = d_Q; p.K = d_K; p.V = d_V;
  p.impl = fused::ImplId::kAuto;

  ASSERT_TRUE(fused::FusedRun(p, nullptr).ok());
  TEST_CUDA(cudaDeviceSynchronize());

  std::vector<__nv_fp8_e5m2> Q_f8(n_out), K_f8(n_out), V_f8(n_out);
  TEST_CUDA(cudaMemcpy(Q_f8.data(), d_Q, n_out, cudaMemcpyDeviceToHost));
  TEST_CUDA(cudaMemcpy(K_f8.data(), d_K, n_out, cudaMemcpyDeviceToHost));
  TEST_CUDA(cudaMemcpy(V_f8.data(), d_V, n_out, cudaMemcpyDeviceToHost));

  std::vector<float> Q_fp32(n_out), K_fp32(n_out), V_fp32(n_out);
  for (size_t i = 0; i < n_out; ++i) {
    Q_fp32[i] = static_cast<float>(Q_f8[i]);
    K_fp32[i] = static_cast<float>(K_f8[i]);
    V_fp32[i] = static_cast<float>(V_f8[i]);
  }

  float tol = 1.0f;
  EXPECT_TRUE(common::CheckEqual(host.Q_ref, Q_fp32, tol))
      << "max_q=" << common::MaxAbsDiff(host.Q_ref, Q_fp32);
  EXPECT_TRUE(common::CheckEqual(host.K_ref, K_fp32, tol))
      << "max_k=" << common::MaxAbsDiff(host.K_ref, K_fp32);
  EXPECT_TRUE(common::CheckEqual(host.V_ref, V_fp32, tol))
      << "max_v=" << common::MaxAbsDiff(host.V_ref, V_fp32);

  cudaFree(d_x); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
}
#endif

TEST(FusedGpuTest, V1RejectsLargeKernel) {
  RequireCudaDevice();
  constexpr int B = 1, L = 128, D = 16, H = 8, k_size = 65;
  std::vector<float> x(static_cast<size_t>(B) * L * D, 0.1f);
  std::vector<float> W_qkv(static_cast<size_t>(3) * H * D, 0.01f);
  std::vector<float> b_qkv(static_cast<size_t>(3) * H, 0.f);
  std::vector<float> W_z(static_cast<size_t>(H) * D, 0.01f);
  std::vector<float> b_z(H, 0.f);
  std::vector<float> K_conv(static_cast<size_t>(k_size) * H, 0.02f);

  DeviceFusedBuffers dev;
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev.x), x.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev.W_qkv), W_qkv.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev.b_qkv), b_qkv.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev.W_z), W_z.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev.b_z), b_z.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev.K_conv), K_conv.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev.Q), static_cast<size_t>(B) * L * H * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev.K), static_cast<size_t>(B) * L * H * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev.V), static_cast<size_t>(B) * L * H * sizeof(float)));
  TEST_CUDA(cudaMemcpy(dev.x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(dev.W_qkv, W_qkv.data(), W_qkv.size() * sizeof(float), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(dev.b_qkv, b_qkv.data(), b_qkv.size() * sizeof(float), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(dev.W_z, W_z.data(), W_z.size() * sizeof(float), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(dev.b_z, b_z.data(), b_z.size() * sizeof(float), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(dev.K_conv, K_conv.data(), K_conv.size() * sizeof(float), cudaMemcpyHostToDevice));

  fused::FusedParams p;
  p.B = B;
  p.L = L;
  p.D = D;
  p.H = H;
  p.k_size = k_size;
  p.dtype = common::DType::kFp32;
  p.x = dev.x;
  p.W_qkv = dev.W_qkv;
  p.b_qkv = dev.b_qkv;
  p.W_z = dev.W_z;
  p.b_z = dev.b_z;
  p.K_conv = dev.K_conv;
  p.Q = dev.Q;
  p.K = dev.K;
  p.V = dev.V;
  p.impl = fused::ImplId::kV1;

  const common::Status st = fused::FusedRun(p, nullptr);
  EXPECT_EQ(st.code, common::StatusCode::kUnsupported);
}

#undef TEST_CUDA
