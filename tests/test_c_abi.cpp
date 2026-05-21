#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <vector>

#include "fused/fused_api.h"
#include "gemm/gemm_api.h"
#include "rmsnorm/rmsnorm_api.h"

TEST(CAbiTest, GemmValidateNullParams) {
  char buf[128] = {};
  EXPECT_EQ(cko_gemm_validate(nullptr, buf, sizeof(buf)), CKO_INVALID_ARGUMENT);
}

TEST(CAbiTest, RmsNormValidateShape) {
  CkoRmsNormParams p{};
  p.rows = 128;
  p.cols = 256;
  p.act_dtype = CKO_DTYPE_FP32;
  p.weight_dtype = CKO_DTYPE_FP32;
  char buf[128] = {};
  EXPECT_EQ(cko_rmsnorm_validate(&p, buf, sizeof(buf)), CKO_OK);
}

TEST(CAbiTest, FusedValidateInvalidK) {
  CkoFusedParams p{};
  p.B = 1;
  p.L = 8;
  p.D = 16;
  p.H = 8;
  p.k_size = 0;
  char buf[128] = {};
  EXPECT_EQ(cko_fused_conv1d_silu_validate(&p, buf, sizeof(buf)), CKO_INVALID_ARGUMENT);
}

TEST(CAbiTest, RmsNormRunFp32Smoke) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  constexpr int rows = 32;
  constexpr int cols = 64;
  std::vector<float> x(static_cast<size_t>(rows) * cols, 0.1f);
  std::vector<float> w(cols, 1.f);
  float *dx = nullptr, *dy = nullptr, *dw = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dx), x.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dy), x.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dw), w.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dw, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

  CkoRmsNormParams p{};
  p.rows = rows;
  p.cols = cols;
  p.act_dtype = CKO_DTYPE_FP32;
  p.weight_dtype = CKO_DTYPE_FP32;
  p.eps = 1e-5f;
  p.input = dx;
  p.weight = dw;
  p.output = dy;
  p.impl = static_cast<int>(rmsnorm::ImplId::kV3);
  EXPECT_EQ(cko_rmsnorm_run(&p, nullptr), CKO_OK);

  cudaFree(dx);
  cudaFree(dy);
  cudaFree(dw);
}

TEST(CAbiTest, FusedRunFp32Smoke) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  constexpr int B = 1, L = 32, D = 16, H = 8, k_size = 4;
  std::vector<float> x(static_cast<size_t>(B) * L * D, 0.1f);
  std::vector<float> W_qkv(static_cast<size_t>(3) * H * D, 0.01f);
  std::vector<float> b_qkv(static_cast<size_t>(3) * H, 0.f);
  std::vector<float> W_z(static_cast<size_t>(H) * D, 0.01f);
  std::vector<float> b_z(H, 0.f);
  std::vector<float> K_conv(static_cast<size_t>(k_size) * H, 0.05f);

  float *dx = nullptr, *dW_qkv = nullptr, *db_qkv = nullptr;
  float *dW_z = nullptr, *db_z = nullptr, *dK = nullptr;
  float *dQ = nullptr, *dKo = nullptr, *dV = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dx), x.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dW_qkv), W_qkv.size() * sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&db_qkv), b_qkv.size() * sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dW_z), W_z.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&db_z), b_z.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dK), K_conv.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dQ), static_cast<size_t>(B) * L * H * sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dKo), static_cast<size_t>(B) * L * H * sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dV), static_cast<size_t>(B) * L * H * sizeof(float)),
            cudaSuccess);

  CkoFusedParams p{};
  p.B = B;
  p.L = L;
  p.D = D;
  p.H = H;
  p.k_size = k_size;
  p.dtype = CKO_DTYPE_FP32;
  p.x = dx;
  p.W_qkv = dW_qkv;
  p.b_qkv = db_qkv;
  p.W_z = dW_z;
  p.b_z = db_z;
  p.K_conv = dK;
  p.Q = dQ;
  p.K = dKo;
  p.V = dV;
  p.impl = static_cast<int>(fused::ImplId::kV3);
  EXPECT_EQ(cko_fused_conv1d_silu_run(&p, nullptr), CKO_OK);

  cudaFree(dx);
  cudaFree(dW_qkv);
  cudaFree(db_qkv);
  cudaFree(dW_z);
  cudaFree(db_z);
  cudaFree(dK);
  cudaFree(dQ);
  cudaFree(dKo);
  cudaFree(dV);
}

TEST(CAbiTest, GemmRunFp32Smoke) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  constexpr int M = 128, N = 128, K = 128;
  std::vector<float> A(static_cast<size_t>(M) * K, 0.1f);
  std::vector<float> B(static_cast<size_t>(K) * N, 0.2f);
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dA), A.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dB), B.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dC), static_cast<size_t>(M) * N * sizeof(float)),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

  CkoGemmParams p{};
  p.M = M;
  p.N = N;
  p.K = K;
  p.lda = K;
  p.ldb = N;
  p.ldc = N;
  p.dtype_a = p.dtype_b = p.dtype_c = CKO_DTYPE_FP32;
  p.A = dA;
  p.B = dB;
  p.C = dC;
  p.impl = static_cast<int>(gemm::ImplId::kV3);
  EXPECT_EQ(cko_gemm_run(&p, nullptr), CKO_OK);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

TEST(CAbiTest, GemmRunBf16Smoke) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  constexpr int M = 128, N = 128, K = 128;
  std::vector<float> A_f(static_cast<size_t>(M) * K, 0.1f);
  std::vector<float> B_f(static_cast<size_t>(K) * N, 0.2f);
  std::vector<__nv_bfloat16> A_bf(A_f.size()), B_bf(B_f.size());
  for (size_t i = 0; i < A_f.size(); ++i) A_bf[i] = __float2bfloat16(A_f[i]);
  for (size_t i = 0; i < B_f.size(); ++i) B_bf[i] = __float2bfloat16(B_f[i]);

  __nv_bfloat16 *dA = nullptr, *dB = nullptr;
  float *dC = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dA), A_bf.size() * sizeof(__nv_bfloat16)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dB), B_bf.size() * sizeof(__nv_bfloat16)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dC), static_cast<size_t>(M) * N * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dA, A_bf.data(), A_bf.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dB, B_bf.data(), B_bf.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), cudaSuccess);

  CkoGemmParams p{};
  p.M = M; p.N = N; p.K = K;
  p.lda = K; p.ldb = N; p.ldc = N;
  p.dtype_a = p.dtype_b = CKO_DTYPE_BF16;
  p.dtype_c = CKO_DTYPE_FP32;
  p.A = dA; p.B = dB; p.C = dC;
  p.impl = static_cast<int>(gemm::ImplId::kCublasBf16);
  EXPECT_EQ(cko_gemm_run(&p, nullptr), CKO_OK);

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

TEST(CAbiTest, GemmRunInt8Smoke) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  constexpr int M = 128, N = 128, K = 128;
  std::vector<int8_t> A_i8(static_cast<size_t>(M) * K, 1);
  std::vector<int8_t> B_i8(static_cast<size_t>(K) * N, 2);

  int8_t *dA = nullptr, *dB = nullptr;
  int32_t* dC = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dA), A_i8.size()), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dB), B_i8.size()), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dC), static_cast<size_t>(M) * N * sizeof(int32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dA, A_i8.data(), A_i8.size(), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dB, B_i8.data(), B_i8.size(), cudaMemcpyHostToDevice), cudaSuccess);

  CkoGemmParams p{};
  p.M = M; p.N = N; p.K = K;
  p.lda = K; p.ldb = N; p.ldc = N;
  p.dtype_a = p.dtype_b = CKO_DTYPE_INT8;
  p.dtype_c = CKO_DTYPE_INT32;
  p.A = dA; p.B = dB; p.C = dC;
  p.impl = static_cast<int>(gemm::ImplId::kCublasInt8);
  EXPECT_EQ(cko_gemm_run(&p, nullptr), CKO_OK);

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

TEST(CAbiTest, GemmRunFp8Smoke) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  constexpr int M = 128, N = 128, K = 128;
  std::vector<__nv_fp8_e4m3> A_f8(static_cast<size_t>(M) * K);
  std::vector<__nv_fp8_e4m3> B_f8(static_cast<size_t>(K) * N);
  for (size_t i = 0; i < A_f8.size(); ++i) A_f8[i] = __nv_fp8_e4m3(0.1f);
  for (size_t i = 0; i < B_f8.size(); ++i) B_f8[i] = __nv_fp8_e4m3(0.2f);

  __nv_fp8_e4m3 *dA = nullptr, *dB = nullptr;
  float *dC = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dA), A_f8.size()), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dB), B_f8.size()), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dC), static_cast<size_t>(M) * N * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dA, A_f8.data(), A_f8.size(), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dB, B_f8.data(), B_f8.size(), cudaMemcpyHostToDevice), cudaSuccess);

  CkoGemmParams p{};
  p.M = M; p.N = N; p.K = K;
  p.lda = K; p.ldb = N; p.ldc = N;
  p.dtype_a = p.dtype_b = CKO_DTYPE_FP8_E4M3;
  p.dtype_c = CKO_DTYPE_FP32;
  p.A = dA; p.B = dB; p.C = dC;
  p.impl = static_cast<int>(gemm::ImplId::kCublasFp8);
  EXPECT_EQ(cko_gemm_run(&p, nullptr), CKO_OK);

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
}
