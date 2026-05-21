#include <gtest/gtest.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "common/benchmark.h"
#include "common/dtype.h"
#include "common/test_case.h"
#include "gemm/gemm_api.h"
#include "gemm/gemm_ref_cpu.h"

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

gemm::ImplId ImplFromString(const std::string& s) {
  if (s == "auto") return gemm::ImplId::kAuto;
  if (s == "v0") return gemm::ImplId::kV0;
  if (s == "v3") return gemm::ImplId::kV3;
  if (s == "fp16") return gemm::ImplId::kFp16;
  if (s == "cublas") return gemm::ImplId::kCublas;
  if (s == "cublasFp16") return gemm::ImplId::kCublasFp16;
  if (s == "cublasBf16") return gemm::ImplId::kCublasBf16;
  if (s == "cublasInt8") return gemm::ImplId::kCublasInt8;
  if (s == "cublasFp8") return gemm::ImplId::kCublasFp8;
  return gemm::ImplId::kAuto;
}

struct HostGemmBuffers {
  std::vector<float> A_f, B_f, C_ref, C_gpu;
  std::vector<__half> A_h, B_h;
  std::vector<__nv_bfloat16> A_bf, B_bf;
  std::vector<int8_t> A_i8, B_i8;
  std::vector<__nv_fp8_e4m3> A_f8e4, B_f8e4;
  std::vector<__nv_fp8_e5m2> A_f8e5, B_f8e5;
};

struct DeviceGemmBuffers {
  void *A = nullptr, *B = nullptr;
  float* C = nullptr;
  ~DeviceGemmBuffers() {
    if (A) cudaFree(A);
    if (B) cudaFree(B);
    if (C) cudaFree(C);
  }
};

void PrepareFp32Case(HostGemmBuffers& h, int M, int N, int K, int seed) {
  h.A_f.assign(static_cast<size_t>(M) * K, 0.f);
  h.B_f.assign(static_cast<size_t>(K) * N, 0.f);
  h.C_ref.assign(static_cast<size_t>(M) * N, 0.f);
  FillRandom(h.A_f, seed);
  FillRandom(h.B_f, seed + 1);

  gemm::GemmParams ref;
  ref.M = M;
  ref.N = N;
  ref.K = K;
  ref.lda = K;
  ref.ldb = N;
  ref.ldc = N;
  ref.A = h.A_f.data();
  ref.B = h.B_f.data();
  ref.C = h.C_ref.data();
  ASSERT_TRUE(gemm::GemmReferenceHost(ref).ok());
}

void UploadFp32Case(const HostGemmBuffers& h, DeviceGemmBuffers& d, int M, int N, int K) {
  TEST_CUDA(cudaMalloc(&d.A, h.A_f.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(&d.B, h.B_f.size() * sizeof(float)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.C), static_cast<size_t>(M) * N * sizeof(float)));
  TEST_CUDA(cudaMemcpy(d.A, h.A_f.data(), h.A_f.size() * sizeof(float), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(d.B, h.B_f.data(), h.B_f.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void PrepareFp16Case(HostGemmBuffers& h, int M, int N, int K, int seed) {
  h.A_f.assign(static_cast<size_t>(M) * K, 0.f);
  h.B_f.assign(static_cast<size_t>(K) * N, 0.f);
  h.C_ref.assign(static_cast<size_t>(M) * N, 0.f);
  FillRandom(h.A_f, seed);
  FillRandom(h.B_f, seed + 1);

  h.A_h.resize(h.A_f.size());
  h.B_h.resize(h.B_f.size());
  for (size_t i = 0; i < h.A_f.size(); ++i) h.A_h[i] = __float2half(h.A_f[i]);
  for (size_t i = 0; i < h.B_f.size(); ++i) h.B_h[i] = __float2half(h.B_f[i]);

  gemm::GemmParams ref;
  ref.M = M;
  ref.N = N;
  ref.K = K;
  ref.lda = K;
  ref.ldb = N;
  ref.ldc = N;
  ref.A = h.A_f.data();
  ref.B = h.B_f.data();
  ref.C = h.C_ref.data();
  ASSERT_TRUE(gemm::GemmReferenceHost(ref).ok());
}

void UploadFp16Case(const HostGemmBuffers& h, DeviceGemmBuffers& d, int M, int N) {
  TEST_CUDA(cudaMalloc(&d.A, h.A_h.size() * sizeof(__half)));
  TEST_CUDA(cudaMalloc(&d.B, h.B_h.size() * sizeof(__half)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.C), static_cast<size_t>(M) * N * sizeof(float)));
  TEST_CUDA(cudaMemcpy(d.A, h.A_h.data(), h.A_h.size() * sizeof(__half), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(d.B, h.B_h.data(), h.B_h.size() * sizeof(__half), cudaMemcpyHostToDevice));
}

void PrepareBf16Case(HostGemmBuffers& h, int M, int N, int K, int seed) {
  h.A_f.assign(static_cast<size_t>(M) * K, 0.f);
  h.B_f.assign(static_cast<size_t>(K) * N, 0.f);
  h.C_ref.assign(static_cast<size_t>(M) * N, 0.f);
  FillRandom(h.A_f, seed);
  FillRandom(h.B_f, seed + 1);

  gemm::GemmParams ref;
  ref.M = M; ref.N = N; ref.K = K;
  ref.lda = K; ref.ldb = N; ref.ldc = N;
  ref.A = h.A_f.data(); ref.B = h.B_f.data(); ref.C = h.C_ref.data();
  ASSERT_TRUE(gemm::GemmReferenceHost(ref).ok());

  h.A_bf.resize(h.A_f.size());
  h.B_bf.resize(h.B_f.size());
  for (size_t i = 0; i < h.A_f.size(); ++i) h.A_bf[i] = __float2bfloat16(h.A_f[i]);
  for (size_t i = 0; i < h.B_f.size(); ++i) h.B_bf[i] = __float2bfloat16(h.B_f[i]);
}

void UploadBf16Case(const HostGemmBuffers& h, DeviceGemmBuffers& d, int M, int N) {
  TEST_CUDA(cudaMalloc(&d.A, h.A_bf.size() * sizeof(__nv_bfloat16)));
  TEST_CUDA(cudaMalloc(&d.B, h.B_bf.size() * sizeof(__nv_bfloat16)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.C), static_cast<size_t>(M) * N * sizeof(float)));
  TEST_CUDA(cudaMemcpy(d.A, h.A_bf.data(), h.A_bf.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(d.B, h.B_bf.data(), h.B_bf.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
}

void PrepareInt8Case(HostGemmBuffers& h, int M, int N, int K, int seed) {
  h.A_f.assign(static_cast<size_t>(M) * K, 0.f);
  h.B_f.assign(static_cast<size_t>(K) * N, 0.f);
  h.C_ref.assign(static_cast<size_t>(M) * N, 0.f);
  FillRandom(h.A_f, seed);
  FillRandom(h.B_f, seed + 1);

  gemm::GemmParams ref;
  ref.M = M; ref.N = N; ref.K = K;
  ref.lda = K;
  ref.ldb = N;
  ref.ldc = N;
  ref.A = h.A_f.data();
  ref.B = h.B_f.data();
  ref.C = h.C_ref.data();
  ASSERT_TRUE(gemm::GemmReferenceHost(ref).ok());

  h.A_i8.resize(h.A_f.size());
  h.B_i8.resize(h.B_f.size());
  for (size_t i = 0; i < h.A_f.size(); ++i)
    h.A_i8[i] = static_cast<int8_t>(std::clamp(h.A_f[i] * 127.f, -128.f, 127.f));
  for (size_t i = 0; i < h.B_f.size(); ++i)
    h.B_i8[i] = static_cast<int8_t>(std::clamp(h.B_f[i] * 127.f, -128.f, 127.f));
}

void UploadInt8Case(const HostGemmBuffers& h, DeviceGemmBuffers& d, int M, int N) {
  TEST_CUDA(cudaMalloc(&d.A, h.A_i8.size() * sizeof(int8_t)));
  TEST_CUDA(cudaMalloc(&d.B, h.B_i8.size() * sizeof(int8_t)));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.C), static_cast<size_t>(M) * N * sizeof(float)));
  TEST_CUDA(cudaMemcpy(d.A, h.A_i8.data(), h.A_i8.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(d.B, h.B_i8.data(), h.B_i8.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
}

void PrepareFp8Case(HostGemmBuffers& h, int M, int N, int K, int seed, common::DType dtype) {
  const bool is_e4m3 = (dtype == common::DType::kFp8E4M3);
  const float scale = is_e4m3 ? (1.f / 448.f) : (1.f / 57344.f);

  h.A_f.assign(static_cast<size_t>(M) * K, 0.f);
  h.B_f.assign(static_cast<size_t>(K) * N, 0.f);
  h.C_ref.assign(static_cast<size_t>(M) * N, 0.f);
  FillRandom(h.A_f, seed);
  FillRandom(h.B_f, seed + 1);

  gemm::GemmParams ref;
  ref.M = M; ref.N = N; ref.K = K;
  ref.lda = K; ref.ldb = N; ref.ldc = N;
  ref.A = h.A_f.data(); ref.B = h.B_f.data(); ref.C = h.C_ref.data();
  ASSERT_TRUE(gemm::GemmReferenceHost(ref).ok());

  if (is_e4m3) {
    h.A_f8e4.resize(h.A_f.size());
    h.B_f8e4.resize(h.B_f.size());
    for (size_t i = 0; i < h.A_f.size(); ++i)
      h.A_f8e4[i] = __nv_fp8_e4m3(h.A_f[i] * scale);
    for (size_t i = 0; i < h.B_f.size(); ++i)
      h.B_f8e4[i] = __nv_fp8_e4m3(h.B_f[i] * scale);
  } else {
    h.A_f8e5.resize(h.A_f.size());
    h.B_f8e5.resize(h.B_f.size());
    for (size_t i = 0; i < h.A_f.size(); ++i)
      h.A_f8e5[i] = __nv_fp8_e5m2(h.A_f[i] * scale);
    for (size_t i = 0; i < h.B_f.size(); ++i)
      h.B_f8e5[i] = __nv_fp8_e5m2(h.B_f[i] * scale);
  }
}

void UploadFp8Case(const HostGemmBuffers& h, DeviceGemmBuffers& d, int M, int N,
                   common::DType dtype) {
  const bool is_e4m3 = (dtype == common::DType::kFp8E4M3);
  const size_t ab_size = h.A_f.size() * (is_e4m3 ? sizeof(__nv_fp8_e4m3) : sizeof(__nv_fp8_e5m2));
  const void* a_data = is_e4m3 ? static_cast<const void*>(h.A_f8e4.data())
                               : static_cast<const void*>(h.A_f8e5.data());
  const void* b_data = is_e4m3 ? static_cast<const void*>(h.B_f8e4.data())
                               : static_cast<const void*>(h.B_f8e5.data());
  TEST_CUDA(cudaMalloc(&d.A, ab_size));
  TEST_CUDA(cudaMalloc(&d.B, ab_size));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d.C), static_cast<size_t>(M) * N * sizeof(float)));
  TEST_CUDA(cudaMemcpy(d.A, a_data, ab_size, cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(d.B, b_data, ab_size, cudaMemcpyHostToDevice));
}

void RunGpuCase(int M, int N, int K, common::DType dtype, gemm::ImplId impl, float tol) {
  HostGemmBuffers host;
  DeviceGemmBuffers dev;
  const int seed = M * 10000 + N * 100 + K;

  gemm::GemmParams p;
  p.M = M;
  p.N = N;
  p.K = K;
  p.lda = K;
  p.ldb = N;
  p.ldc = N;
  p.impl = impl;
  p.alignment_policy = gemm::AlignmentPolicy::kFallback;

  if (dtype == common::DType::kFp32) {
    PrepareFp32Case(host, M, N, K, seed);
    UploadFp32Case(host, dev, M, N, K);
    p.dtype_a = p.dtype_b = p.dtype_c = common::DType::kFp32;
    p.A = dev.A;
    p.B = dev.B;
    p.C = dev.C;
  } else if (dtype == common::DType::kFp16) {
    PrepareFp16Case(host, M, N, K, seed);
    UploadFp16Case(host, dev, M, N);
    p.dtype_a = p.dtype_b = common::DType::kFp16;
    p.dtype_c = common::DType::kFp32;
    p.A = dev.A;
    p.B = dev.B;
    p.C = dev.C;
  } else if (dtype == common::DType::kBf16) {
    PrepareBf16Case(host, M, N, K, seed);
    UploadBf16Case(host, dev, M, N);
    p.dtype_a = p.dtype_b = common::DType::kBf16;
    p.dtype_c = common::DType::kFp32;
    p.A = dev.A;
    p.B = dev.B;
    p.C = dev.C;
  } else if (dtype == common::DType::kInt8) {
    PrepareInt8Case(host, M, N, K, seed);
    UploadInt8Case(host, dev, M, N);
    p.dtype_a = p.dtype_b = common::DType::kInt8;
    p.dtype_c = common::DType::kFp32;
    p.A = dev.A;
    p.B = dev.B;
    p.C = dev.C;
  } else if (dtype == common::DType::kFp8E4M3 || dtype == common::DType::kFp8E5M2) {
    PrepareFp8Case(host, M, N, K, seed, dtype);
    UploadFp8Case(host, dev, M, N, dtype);
    p.dtype_a = p.dtype_b = dtype;
    p.dtype_c = common::DType::kFp32;
    p.A = dev.A;
    p.B = dev.B;
    p.C = dev.C;
  }

  ASSERT_TRUE(gemm::GemmRun(p, nullptr).ok());
  TEST_CUDA(cudaDeviceSynchronize());

  std::vector<float> C_gpu(static_cast<size_t>(M) * N);
  TEST_CUDA(cudaMemcpy(C_gpu.data(), dev.C, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
  EXPECT_TRUE(common::CheckEqual(host.C_ref, C_gpu, tol)) << "max diff "
                                                          << common::MaxAbsDiff(host.C_ref, C_gpu);
}

}  // namespace

TEST(GemmGpuTest, Fp32SmokeV3) {
  RequireCudaDevice();
  RunGpuCase(128, 128, 128, common::DType::kFp32, gemm::ImplId::kV3, 1e-3f);
}

TEST(GemmGpuTest, Fp32LegacyV0) {
  RequireCudaDevice();
  RunGpuCase(64, 64, 64, common::DType::kFp32, gemm::ImplId::kV0, 1e-3f);
}

TEST(GemmGpuTest, Fp32AutoFallbackCublas) {
  RequireCudaDevice();
  RunGpuCase(100, 200, 300, common::DType::kFp32, gemm::ImplId::kAuto, 1e-3f);
}

TEST(GemmGpuTest, CublasFp32Smoke) {
  RequireCudaDevice();
  RunGpuCase(128, 128, 128, common::DType::kFp32, gemm::ImplId::kCublas, 1e-3f);
}

TEST(GemmGpuTest, Fp16Aligned) {
  RequireCudaDevice();
  RunGpuCase(256, 256, 256, common::DType::kFp16, gemm::ImplId::kFp16, 5e-2f);
}

TEST(GemmGpuTest, CublasFp16Smoke) {
  RequireCudaDevice();
  RunGpuCase(128, 128, 128, common::DType::kFp16, gemm::ImplId::kCublasFp16, 2.0f);
}

namespace {

void SmokeGpuCase(int M, int N, int K, common::DType dtype_a, common::DType dtype_b,
                  common::DType dtype_c, gemm::ImplId impl, size_t elem_a, size_t elem_b,
                  size_t elem_c) {
  std::vector<char> A_buf(elem_a, 0), B_buf(elem_b, 0);
  for (size_t i = 0; i < elem_a; ++i) A_buf[i] = static_cast<char>(i % 127 + 1);
  for (size_t i = 0; i < elem_b; ++i) B_buf[i] = static_cast<char>((i + 1) % 127 + 1);

  DeviceGemmBuffers dev;
  TEST_CUDA(cudaMalloc(&dev.A, elem_a));
  TEST_CUDA(cudaMalloc(&dev.B, elem_b));
  TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev.C), elem_c));
  TEST_CUDA(cudaMemcpy(dev.A, A_buf.data(), elem_a, cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(dev.B, B_buf.data(), elem_b, cudaMemcpyHostToDevice));

  gemm::GemmParams p;
  p.M = M; p.N = N; p.K = K;
  p.lda = K; p.ldb = N; p.ldc = N;
  p.dtype_a = dtype_a; p.dtype_b = dtype_b; p.dtype_c = dtype_c;
  p.A = dev.A; p.B = dev.B; p.C = dev.C;
  p.impl = impl;
  p.alignment_policy = gemm::AlignmentPolicy::kFallback;
  ASSERT_TRUE(gemm::GemmRun(p, nullptr).ok());
  TEST_CUDA(cudaDeviceSynchronize());
}

}  // namespace

TEST(GemmGpuTest, Bf16Aligned) {
  RequireCudaDevice();
  RunGpuCase(128, 128, 128, common::DType::kBf16, gemm::ImplId::kCublasBf16, 0.5f);
}

TEST(GemmGpuTest, Fp8E4M3Aligned) {
  RequireCudaDevice();
  RunGpuCase(128, 128, 128, common::DType::kFp8E4M3, gemm::ImplId::kCublasFp8, 500.f);
}

TEST(GemmGpuTest, Int8Smoke) {
  RequireCudaDevice();
  SmokeGpuCase(128, 128, 128, common::DType::kInt8, common::DType::kInt8,
               common::DType::kInt32, gemm::ImplId::kCublasInt8,
               static_cast<size_t>(128) * 128, static_cast<size_t>(128) * 128,
               static_cast<size_t>(128) * 128 * sizeof(int32_t));
}

TEST(GemmGpuTest, Fp8E5M2Smoke) {
  GTEST_SKIP() << "fp8e5m2 cublasLt matmul not available";
}

TEST(GemmGpuTest, CatalogPassCases) {
  RequireCudaDevice();
  const auto cat = common::LoadTestCaseCatalog(
      std::string(CKO_SOURCE_DIR) + "/configs/test_cases/gemm.json");
  for (const auto& c : cat.cases) {
    if (c.expect != common::ExpectStatus::kPass) continue;

    common::DType dtype = common::DType::kFp32;
    ASSERT_TRUE(common::ParseDType(c.gemm.dtype.c_str(), &dtype));
    float tol = 1e-3f;
    if (dtype == common::DType::kFp16) tol = 5e-2f;
    else if (dtype == common::DType::kBf16) tol = 0.5f;
    else if (dtype == common::DType::kInt8) tol = 50.f;
    else if (dtype == common::DType::kFp8E4M3 || dtype == common::DType::kFp8E5M2) tol = 500.f;
    RunGpuCase(c.gemm.M, c.gemm.N, c.gemm.K, dtype, ImplFromString(c.gemm.impl), tol);
  }
}

#undef TEST_CUDA
