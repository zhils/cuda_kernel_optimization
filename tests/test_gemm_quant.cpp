#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"
#include "common/dtype.h"
#include "gemm/gemm_api.h"
#include "gemm/gemm_quant.h"

#define TEST_CUDA(call) ASSERT_EQ(call, cudaSuccess)

namespace {

void RequireCudaDevice() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
    GTEST_SKIP() << "No CUDA device available";
  }
}

void CpuGemm(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0;
      for (int k = 0; k < K; ++k)
        sum += static_cast<double>(A[i * K + k]) * static_cast<double>(B[k * N + j]);
      C[i * N + j] = static_cast<float>(sum);
    }
  }
}

struct QuantTestCase {
  const char* name;
  int M, N, K;
  common::DType quant_dtype;
  gemm::QuantScheme scheme_a;
  gemm::QuantScheme scheme_b;
  double tol;
};

void RunQuantTest(const QuantTestCase& tc) {
  const int M = tc.M, N = tc.N, K = tc.K;
  const size_t n_a = static_cast<size_t>(M) * K;
  const size_t n_b = static_cast<size_t>(K) * N;
  const size_t n_c = static_cast<size_t>(M) * N;

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  std::vector<float> A_fp32(n_a), B_fp32(n_b), C_ref(n_c);
  for (auto& v : A_fp32) v = dist(gen);
  for (auto& v : B_fp32) v = dist(gen);

  CpuGemm(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  TEST_CUDA(cudaMalloc(&d_A, n_a * sizeof(float)));
  TEST_CUDA(cudaMalloc(&d_B, n_b * sizeof(float)));
  TEST_CUDA(cudaMalloc(&d_C, n_c * sizeof(float)));
  TEST_CUDA(cudaMemcpy(d_A, A_fp32.data(), n_a * sizeof(float), cudaMemcpyHostToDevice));
  TEST_CUDA(cudaMemcpy(d_B, B_fp32.data(), n_b * sizeof(float), cudaMemcpyHostToDevice));

  common::Status st = gemm::GemmQuantizedRun(
      d_A, d_B, d_C, M, N, K, tc.quant_dtype, tc.scheme_a, tc.scheme_b, nullptr);
  ASSERT_TRUE(st.ok()) << tc.name << ": " << st.message;

  TEST_CUDA(cudaDeviceSynchronize());

  std::vector<float> C_gpu(n_c);
  TEST_CUDA(cudaMemcpy(C_gpu.data(), d_C, n_c * sizeof(float), cudaMemcpyDeviceToHost));

  TEST_CUDA(cudaFree(d_A)); TEST_CUDA(cudaFree(d_B)); TEST_CUDA(cudaFree(d_C));

  double max_diff = common::MaxAbsDiff(C_ref, C_gpu);
  EXPECT_LT(max_diff, tc.tol) << tc.name << ": max_diff=" << max_diff;
}

}  // namespace

TEST(GemmQuantTest, PerTensorInt8) {
  RequireCudaDevice();
  RunQuantTest({"per_tensor_int8", 128, 128, 128, common::DType::kInt8,
                gemm::QuantScheme::kPerTensor, gemm::QuantScheme::kPerTensor, 20.0});
}

TEST(GemmQuantTest, PerRowInt8) {
  RequireCudaDevice();
  RunQuantTest({"per_row_int8", 128, 128, 128, common::DType::kInt8,
                gemm::QuantScheme::kPerRow, gemm::QuantScheme::kPerRow, 25.0});
}

TEST(GemmQuantTest, PerTensorFp8E4M3) {
  RequireCudaDevice();
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
  RunQuantTest({"per_tensor_fp8e4m3", 128, 128, 128, common::DType::kFp8E4M3,
                gemm::QuantScheme::kPerTensor, gemm::QuantScheme::kPerTensor, 10.0});
#else
  GTEST_SKIP() << "FP8 not available with this CUDA toolchain";
#endif
}

TEST(GemmQuantTest, PerRowFp8E4M3) {
  RequireCudaDevice();
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
  RunQuantTest({"per_row_fp8e4m3", 128, 128, 128, common::DType::kFp8E4M3,
                gemm::QuantScheme::kPerRow, gemm::QuantScheme::kPerRow, 12.0});
#else
  GTEST_SKIP() << "FP8 not available with this CUDA toolchain";
#endif
}

TEST(GemmQuantTest, PerTensorFp8E5M2) {
  RequireCudaDevice();
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
  RunQuantTest({"per_tensor_fp8e5m2", 128, 128, 128, common::DType::kFp8E5M2,
                gemm::QuantScheme::kPerTensor, gemm::QuantScheme::kPerTensor, 15.0});
#else
  GTEST_SKIP() << "FP8 not available with this CUDA toolchain";
#endif
}

TEST(GemmQuantTest, MixedSchemeARowBTensor) {
  RequireCudaDevice();
  RunQuantTest({"mixed_row_tensor", 128, 128, 128, common::DType::kInt8,
                gemm::QuantScheme::kPerRow, gemm::QuantScheme::kPerTensor, 25.0});
}

TEST(GemmQuantTest, AlignmentFallbackToFp32) {
  RequireCudaDevice();
  RunQuantTest({"unaligned_fallback", 100, 100, 100, common::DType::kInt8,
                gemm::QuantScheme::kPerTensor, gemm::QuantScheme::kPerTensor, 1e-3});
}

TEST(GemmQuantTest, LargeSquare) {
  RequireCudaDevice();
  RunQuantTest({"large_square", 512, 512, 512, common::DType::kInt8,
                gemm::QuantScheme::kPerTensor, gemm::QuantScheme::kPerTensor, 25.0});
}

TEST(GemmQuantTest, Rectangular) {
  RequireCudaDevice();
  RunQuantTest({"rectangular", 256, 128, 512, common::DType::kInt8,
                gemm::QuantScheme::kPerRow, gemm::QuantScheme::kPerRow, 30.0});
}
