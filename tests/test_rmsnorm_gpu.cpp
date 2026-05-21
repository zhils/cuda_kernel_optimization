#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

#include "common/benchmark.h"
#include "common/dtype.h"
#include "common/test_case.h"
#include "rmsnorm/rmsnorm_api.h"
#include "rmsnorm/rmsnorm_quant.h"
#include "rmsnorm/rmsnorm_ref_cpu.h"

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

void FillDeterministicFp32(std::vector<float>& x, int rows, int cols, float scale) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      x[static_cast<size_t>(r) * cols + c] =
          scale * (0.01f * static_cast<float>(r + 1) + 0.001f * static_cast<float>(c + 1));
    }
  }
}

struct HostBuffers {
  std::vector<float> x_f;
  std::vector<float> w_f;
  std::vector<float> y_ref_fp32;
  std::vector<__half> x_h, w_h, y_h;
  std::vector<__nv_bfloat16> x_bf, w_bf, y_bf;
  std::vector<uint8_t> x_q, y_q;
  std::vector<float> x_scale, y_scale;
};

struct DeviceBuffers {
  void* input = nullptr;
  void* weight = nullptr;
  void* output = nullptr;
  float* input_scale = nullptr;
  float* output_scale = nullptr;

  ~DeviceBuffers() {
    if (input) cudaFree(input);
    if (weight) cudaFree(weight);
    if (output) cudaFree(output);
    if (input_scale) cudaFree(input_scale);
    if (output_scale) cudaFree(output_scale);
  }
};

rmsnorm::RmsNormParams BindHostParams(HostBuffers& h, int rows, int cols, common::DType act,
                                      common::DType wdt) {
  rmsnorm::RmsNormParams p;
  p.rows = rows;
  p.cols = cols;
  p.act_dtype = act;
  p.weight_dtype = wdt;
  p.eps = 1e-5f;

  if (act == common::DType::kFp32) {
    p.input = h.x_f.data();
    p.weight = h.w_f.data();
    p.output = h.y_ref_fp32.data();
  } else if (act == common::DType::kFp16) {
    p.input = h.x_h.data();
    p.weight = h.w_h.data();
    p.output = h.y_h.data();
  } else if (act == common::DType::kBf16) {
    p.input = h.x_bf.data();
    p.weight = h.w_bf.data();
    p.output = h.y_bf.data();
  } else {
    p.input = h.x_q.data();
    p.weight = h.w_f.data();
    p.output = h.y_q.data();
    p.input_scale = h.x_scale.data();
    p.output_scale = h.y_scale.data();
  }
  return p;
}

void PrepareHostCase(HostBuffers& h, int rows, int cols, common::DType act, common::DType wdt) {
  const size_t n = static_cast<size_t>(rows) * cols;
  h.x_f.assign(n, 0.f);
  h.w_f.assign(cols, 0.f);
  FillDeterministicFp32(h.x_f, rows, cols, 0.4f);
  for (int c = 0; c < cols; ++c) {
    h.w_f[c] = 0.9f + 0.002f * static_cast<float>(c);
  }

  if (act == common::DType::kFp32) {
    h.y_ref_fp32.assign(n, 0.f);
  } else if (act == common::DType::kFp16) {
    h.x_h.resize(n);
    h.w_h.resize(cols);
    h.y_h.assign(n, __float2half(0.f));
    for (size_t i = 0; i < n; ++i) h.x_h[i] = __float2half(h.x_f[i]);
    for (int c = 0; c < cols; ++c) h.w_h[c] = __float2half(h.w_f[c]);
  } else if (act == common::DType::kBf16) {
    h.x_bf.resize(n);
    h.w_bf.resize(cols);
    h.y_bf.assign(n, __float2bfloat16(0.f));
    for (size_t i = 0; i < n; ++i) h.x_bf[i] = __float2bfloat16(h.x_f[i]);
    for (int c = 0; c < cols; ++c) h.w_bf[c] = __float2bfloat16(h.w_f[c]);
  } else {
    rmsnorm::ActivationDtype adt = rmsnorm::ActivationDtype::kInt8;
    if (act == common::DType::kFp8E4M3) adt = rmsnorm::ActivationDtype::kFp8E4M3;
    if (act == common::DType::kFp8E5M2) adt = rmsnorm::ActivationDtype::kFp8E5M2;
    rmsnorm::QuantizeActivationHost(adt, h.x_f, rows, cols, h.x_q, h.x_scale);
    h.y_q.assign(h.x_q.size(), 0);
    h.y_scale.assign(rows, 0.f);
  }

  const auto ref_p = BindHostParams(h, rows, cols, act, wdt);
  ASSERT_TRUE(rmsnorm::RmsNormReferenceHost(ref_p).ok());
}

void UploadCase(const HostBuffers& h, DeviceBuffers& d, int rows, int cols, common::DType act,
                common::DType wdt) {
  const size_t n = static_cast<size_t>(rows) * cols;
  const size_t act_bytes = n * common::DTypeBytes(act);
  const size_t w_bytes = static_cast<size_t>(cols) * common::DTypeBytes(wdt);

  TEST_CUDA(cudaMalloc(&d.input, act_bytes));
  TEST_CUDA(cudaMalloc(&d.weight, w_bytes));
  TEST_CUDA(cudaMalloc(&d.output, act_bytes));

  if (act == common::DType::kFp32) {
    TEST_CUDA(cudaMemcpy(d.input, h.x_f.data(), act_bytes, cudaMemcpyHostToDevice));
  } else if (act == common::DType::kFp16) {
    TEST_CUDA(cudaMemcpy(d.input, h.x_h.data(), act_bytes, cudaMemcpyHostToDevice));
  } else if (act == common::DType::kBf16) {
    TEST_CUDA(cudaMemcpy(d.input, h.x_bf.data(), act_bytes, cudaMemcpyHostToDevice));
  } else {
    TEST_CUDA(cudaMemcpy(d.input, h.x_q.data(), act_bytes, cudaMemcpyHostToDevice));
    TEST_CUDA(cudaMalloc(&d.input_scale, rows * sizeof(float)));
    TEST_CUDA(cudaMalloc(&d.output_scale, rows * sizeof(float)));
    TEST_CUDA(
        cudaMemcpy(d.input_scale, h.x_scale.data(), rows * sizeof(float), cudaMemcpyHostToDevice));
  }

  if (wdt == common::DType::kFp32) {
    TEST_CUDA(cudaMemcpy(d.weight, h.w_f.data(), w_bytes, cudaMemcpyHostToDevice));
  } else if (wdt == common::DType::kFp16) {
    TEST_CUDA(cudaMemcpy(d.weight, h.w_h.data(), w_bytes, cudaMemcpyHostToDevice));
  } else {
    TEST_CUDA(cudaMemcpy(d.weight, h.w_bf.data(), w_bytes, cudaMemcpyHostToDevice));
  }
}

std::vector<float> DequantizeHostOutput(const HostBuffers& /*host*/, int rows, int cols,
                                        common::DType act, const void* output,
                                        const float* output_scale) {
  std::vector<float> out;
  if (!rmsnorm::DequantizeOutputToFp32(act, output, output_scale, rows, cols, out).ok()) {
    ADD_FAILURE() << "DequantizeOutputToFp32 failed";
  }
  return out;
}

void RunGpuCase(int rows, int cols, common::DType act, common::DType wdt,
                rmsnorm::ImplId impl = rmsnorm::ImplId::kAuto) {
  HostBuffers host;
  DeviceBuffers dev;
  PrepareHostCase(host, rows, cols, act, wdt);
  UploadCase(host, dev, rows, cols, act, wdt);

  const void* ref_out = nullptr;
  const float* ref_scale = nullptr;
  if (act == common::DType::kFp32) {
    ref_out = host.y_ref_fp32.data();
  } else if (act == common::DType::kFp16) {
    ref_out = host.y_h.data();
  } else if (act == common::DType::kBf16) {
    ref_out = host.y_bf.data();
  } else {
    ref_out = host.y_q.data();
    ref_scale = host.y_scale.data();
  }
  const std::vector<float> y_ref =
      DequantizeHostOutput(host, rows, cols, act, ref_out, ref_scale);

  rmsnorm::RmsNormParams p;
  p.rows = rows;
  p.cols = cols;
  p.act_dtype = act;
  p.weight_dtype = wdt;
  p.eps = 1e-5f;
  p.input = dev.input;
  p.weight = dev.weight;
  p.output = dev.output;
  p.input_scale = dev.input_scale;
  p.output_scale = dev.output_scale;
  p.impl = impl;

  ASSERT_TRUE(rmsnorm::RmsNormRun(p, nullptr).ok());
  TEST_CUDA(cudaDeviceSynchronize());

  const size_t n = static_cast<size_t>(rows) * cols;
  std::vector<float> y_gpu_raw_fp32(n);
  std::vector<__half> y_gpu_h;
  std::vector<__nv_bfloat16> y_gpu_bf;
  std::vector<uint8_t> y_gpu_q;
  std::vector<float> y_gpu_scale(rows);

  const void* gpu_out = nullptr;
  const float* gpu_scale = nullptr;
  if (act == common::DType::kFp32) {
    TEST_CUDA(cudaMemcpy(y_gpu_raw_fp32.data(), dev.output, n * sizeof(float),
                         cudaMemcpyDeviceToHost));
    gpu_out = y_gpu_raw_fp32.data();
  } else if (act == common::DType::kFp16) {
    y_gpu_h.resize(n);
    TEST_CUDA(cudaMemcpy(y_gpu_h.data(), dev.output, n * sizeof(__half), cudaMemcpyDeviceToHost));
    gpu_out = y_gpu_h.data();
  } else if (act == common::DType::kBf16) {
    y_gpu_bf.resize(n);
    TEST_CUDA(
        cudaMemcpy(y_gpu_bf.data(), dev.output, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    gpu_out = y_gpu_bf.data();
  } else {
    y_gpu_q.resize(n);
    TEST_CUDA(cudaMemcpy(y_gpu_q.data(), dev.output, n, cudaMemcpyDeviceToHost));
    TEST_CUDA(cudaMemcpy(y_gpu_scale.data(), dev.output_scale, rows * sizeof(float),
                         cudaMemcpyDeviceToHost));
    gpu_out = y_gpu_q.data();
    gpu_scale = y_gpu_scale.data();
  }

  const std::vector<float> y_gpu =
      DequantizeHostOutput(host, rows, cols, act, gpu_out, gpu_scale);
  EXPECT_TRUE(common::CheckEqual(y_ref, y_gpu, rmsnorm::DefaultAbsTolerance(act)));
}

}  // namespace

TEST(RmsNormGpuTest, Fp32SmokeV3) {
  RequireCudaDevice();
  RunGpuCase(128, 128, common::DType::kFp32, common::DType::kFp32, rmsnorm::ImplId::kV3);
}

TEST(RmsNormGpuTest, Fp32LegacyV0) {
  RequireCudaDevice();
  RunGpuCase(64, 64, common::DType::kFp32, common::DType::kFp32, rmsnorm::ImplId::kV0);
}

TEST(RmsNormGpuTest, Fp16Act) {
  RequireCudaDevice();
  RunGpuCase(128, 256, common::DType::kFp16, common::DType::kFp16);
}

TEST(RmsNormGpuTest, Int8Quantized) {
  RequireCudaDevice();
  RunGpuCase(64, 128, common::DType::kInt8, common::DType::kFp32);
}

TEST(RmsNormGpuTest, CatalogPassCases) {
  RequireCudaDevice();
  const auto cat = common::LoadTestCaseCatalog(
      std::string(CKO_SOURCE_DIR) + "/configs/test_cases/rmsnorm.json");
  for (const auto& c : cat.cases) {
    if (c.expect != common::ExpectStatus::kPass) continue;

    common::DType act = common::DType::kFp32;
    common::DType wdt = common::DType::kFp32;
    common::ParseDType(c.rmsnorm.act_dtype.c_str(), &act);
    common::ParseDType(c.rmsnorm.weight_dtype.c_str(), &wdt);
    RunGpuCase(c.rmsnorm.rows, c.rmsnorm.cols, act, wdt);
  }
}

TEST(RmsNormGpuTest, LegacyImplRejectsFp16) {
  RequireCudaDevice();
  HostBuffers host;
  DeviceBuffers dev;
  PrepareHostCase(host, 32, 32, common::DType::kFp16, common::DType::kFp16);
  UploadCase(host, dev, 32, 32, common::DType::kFp16, common::DType::kFp16);

  rmsnorm::RmsNormParams p;
  p.rows = 32;
  p.cols = 32;
  p.act_dtype = common::DType::kFp16;
  p.weight_dtype = common::DType::kFp16;
  p.input = dev.input;
  p.weight = dev.weight;
  p.output = dev.output;
  p.impl = rmsnorm::ImplId::kV1;

  const common::Status st = rmsnorm::RmsNormRun(p, nullptr);
  EXPECT_EQ(st.code, common::StatusCode::kUnsupported);
}

#undef TEST_CUDA
