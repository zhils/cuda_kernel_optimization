#include <gtest/gtest.h>

#include <cmath>
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

void FillDeterministicFp32(std::vector<float>& x, int rows, int cols, float scale) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      x[static_cast<size_t>(r) * cols + c] =
          scale * (0.01f * static_cast<float>(r + 1) + 0.001f * static_cast<float>(c + 1));
    }
  }
}

rmsnorm::RmsNormParams MakeFp32Params(int rows, int cols, std::vector<float>& x,
                                      std::vector<float>& w, std::vector<float>& y) {
  rmsnorm::RmsNormParams p;
  p.rows = rows;
  p.cols = cols;
  p.act_dtype = common::DType::kFp32;
  p.weight_dtype = common::DType::kFp32;
  p.eps = 1e-5f;
  p.input = x.data();
  p.weight = w.data();
  p.output = y.data();
  return p;
}

}  // namespace

TEST(RmsNormRefCpuTest, Fp32KnownValues) {
  constexpr int rows = 2;
  constexpr int cols = 4;
  std::vector<float> x = {1.f, 2.f, 3.f, 4.f, 1.f, 1.f, 1.f, 1.f};
  std::vector<float> w = {1.f, 1.f, 1.f, 1.f};
  std::vector<float> y(cols * rows, 0.f);

  const auto p = MakeFp32Params(rows, cols, x, w, y);
  ASSERT_TRUE(rmsnorm::RmsNormReferenceHost(p).ok());

  std::vector<float> y2(cols * rows, 0.f);
  rmsnorm::RmsNormFp32Core(x.data(), w.data(), y2.data(), rows, cols, p.eps);
  EXPECT_TRUE(common::CheckEqual(y, y2, 1e-6f));
}

TEST(RmsNormRefCpuTest, Fp32DeterministicFinite) {
  constexpr int rows = 16;
  constexpr int cols = 32;
  std::vector<float> x(static_cast<size_t>(rows) * cols);
  std::vector<float> w(cols, 1.f);
  std::vector<float> y(static_cast<size_t>(rows) * cols);
  FillDeterministicFp32(x, rows, cols, 0.5f);

  const auto p = MakeFp32Params(rows, cols, x, w, y);
  ASSERT_TRUE(rmsnorm::RmsNormReferenceHost(p).ok());

  for (float v : y) {
    EXPECT_TRUE(std::isfinite(v));
  }
}

TEST(RmsNormRefCpuTest, Fp16RoundTrip) {
  constexpr int rows = 8;
  constexpr int cols = 16;
  std::vector<float> x_f(static_cast<size_t>(rows) * cols);
  std::vector<float> w_f(cols);
  FillDeterministicFp32(x_f, rows, cols, 1.f);
  for (int c = 0; c < cols; ++c) w_f[c] = 0.8f + 0.01f * c;

  std::vector<__half> x_h(x_f.size());
  std::vector<__half> w_h(cols);
  std::vector<__half> y_h(x_f.size());
  for (size_t i = 0; i < x_f.size(); ++i) x_h[i] = __float2half(x_f[i]);
  for (int c = 0; c < cols; ++c) w_h[c] = __float2half(w_f[c]);

  rmsnorm::RmsNormParams p;
  p.rows = rows;
  p.cols = cols;
  p.act_dtype = common::DType::kFp16;
  p.weight_dtype = common::DType::kFp16;
  p.input = x_h.data();
  p.weight = w_h.data();
  p.output = y_h.data();
  ASSERT_TRUE(rmsnorm::RmsNormReferenceHost(p).ok());

  std::vector<float> y_dequant;
  ASSERT_TRUE(
      rmsnorm::DequantizeOutputToFp32(common::DType::kFp16, y_h.data(), nullptr, rows, cols,
                                      y_dequant)
          .ok());

  std::vector<float> y_ref(static_cast<size_t>(rows) * cols);
  rmsnorm::RmsNormFp32Core(x_f.data(), w_f.data(), y_ref.data(), rows, cols, p.eps);
  for (size_t i = 0; i < y_ref.size(); ++i) {
    y_ref[i] = __half2float(__float2half(y_ref[i]));
  }
  EXPECT_TRUE(common::CheckEqual(y_dequant, y_ref, rmsnorm::DefaultAbsTolerance(common::DType::kFp16)));
}

TEST(RmsNormRefCpuTest, Int8QuantizedPath) {
  constexpr int rows = 4;
  constexpr int cols = 8;
  std::vector<float> x_f(static_cast<size_t>(rows) * cols);
  std::vector<float> w_f(cols, 1.f);
  FillDeterministicFp32(x_f, rows, cols, 2.f);

  std::vector<uint8_t> x_q;
  std::vector<float> x_scale;
  rmsnorm::QuantizeActivationHost(rmsnorm::ActivationDtype::kInt8, x_f, rows, cols, x_q,
                                  x_scale);

  std::vector<uint8_t> y_q(x_q.size());
  std::vector<float> y_scale(rows);

  rmsnorm::RmsNormParams p;
  p.rows = rows;
  p.cols = cols;
  p.act_dtype = common::DType::kInt8;
  p.weight_dtype = common::DType::kFp32;
  p.input = x_q.data();
  p.weight = w_f.data();
  p.output = y_q.data();
  p.input_scale = x_scale.data();
  p.output_scale = y_scale.data();
  ASSERT_TRUE(rmsnorm::RmsNormReferenceHost(p).ok());

  std::vector<uint8_t> y_q_ref;
  std::vector<float> y_scale_ref;
  rmsnorm::RMSNormQuantizedCPU(rmsnorm::ActivationDtype::kInt8, x_q, x_scale, w_f, y_q_ref,
                               y_scale_ref, rows, cols, p.eps);

  const auto y_a = rmsnorm::DequantizeMatrixHost(rmsnorm::ActivationDtype::kInt8, y_q, y_scale,
                                                 rows, cols);
  const auto y_b = rmsnorm::DequantizeMatrixHost(rmsnorm::ActivationDtype::kInt8, y_q_ref,
                                                 y_scale_ref, rows, cols);
  EXPECT_TRUE(common::CheckEqual(y_a, y_b, rmsnorm::DefaultAbsTolerance(common::DType::kInt8)));
}

TEST(RmsNormRefCpuTest, CatalogCasesReferenceOk) {
  const auto cat = common::LoadTestCaseCatalog(
      std::string(CKO_SOURCE_DIR) + "/configs/test_cases/rmsnorm.json");
  for (const auto& c : cat.cases) {
    if (c.expect != common::ExpectStatus::kPass) continue;

    const int rows = c.rmsnorm.rows;
    const int cols = c.rmsnorm.cols;
    common::DType act = common::DType::kFp32;
    common::DType wdt = common::DType::kFp32;
    common::ParseDType(c.rmsnorm.act_dtype.c_str(), &act);
    common::ParseDType(c.rmsnorm.weight_dtype.c_str(), &wdt);

    std::vector<float> x_f(static_cast<size_t>(rows) * cols);
    std::vector<float> w_f(cols);
    FillDeterministicFp32(x_f, rows, cols, 0.3f);
    for (int i = 0; i < cols; ++i) w_f[i] = 1.f;

    rmsnorm::RmsNormParams p;
    p.rows = rows;
    p.cols = cols;
    p.act_dtype = act;
    p.weight_dtype = wdt;
    p.eps = 1e-5f;

    std::vector<float> x_store, w_store, y_store;
    std::vector<__half> x_h, w_h, y_h;
    std::vector<__nv_bfloat16> x_bf, w_bf, y_bf;
    std::vector<uint8_t> x_q, y_q;
    std::vector<float> x_scale, y_scale;

    if (act == common::DType::kFp32) {
      y_store.assign(x_f.size(), 0.f);
      p.input = x_f.data();
      p.weight = w_f.data();
      p.output = y_store.data();
    } else if (act == common::DType::kFp16) {
      x_h.resize(x_f.size());
      w_h.resize(cols);
      y_h.resize(x_f.size());
      for (size_t i = 0; i < x_f.size(); ++i) x_h[i] = __float2half(x_f[i]);
      for (int i = 0; i < cols; ++i) w_h[i] = __float2half(w_f[static_cast<size_t>(i)]);
      p.input = x_h.data();
      p.weight = w_h.data();
      p.output = y_h.data();
    } else if (act == common::DType::kBf16) {
      x_bf.resize(x_f.size());
      w_bf.resize(cols);
      y_bf.resize(x_f.size());
      for (size_t i = 0; i < x_f.size(); ++i) x_bf[i] = __float2bfloat16(x_f[i]);
      for (int i = 0; i < cols; ++i) {
        w_bf[i] = __float2bfloat16(w_f[static_cast<size_t>(i)]);
      }
      p.input = x_bf.data();
      p.weight = w_bf.data();
      p.output = y_bf.data();
    } else {
      rmsnorm::ActivationDtype adt = rmsnorm::ActivationDtype::kInt8;
      if (act == common::DType::kFp8E4M3) adt = rmsnorm::ActivationDtype::kFp8E4M3;
      if (act == common::DType::kFp8E5M2) adt = rmsnorm::ActivationDtype::kFp8E5M2;
      rmsnorm::QuantizeActivationHost(adt, x_f, rows, cols, x_q, x_scale);
      y_q.assign(x_q.size(), 0);
      y_scale.assign(rows, 0.f);
      p.input = x_q.data();
      p.weight = w_f.data();
      p.output = y_q.data();
      p.input_scale = x_scale.data();
      p.output_scale = y_scale.data();
    }

    const common::Status st = rmsnorm::RmsNormReferenceHost(p);
    EXPECT_TRUE(st.ok()) << c.id << " msg=" << st.message;
  }
}
