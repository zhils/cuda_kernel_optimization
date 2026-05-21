#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"
#include "rmsnorm/rmsnorm_api.h"
#include "rmsnorm/rmsnorm_quant.h"
#include "rmsnorm/rmsnorm_ref_cpu.h"

namespace {

constexpr int kWarmup = 3;
constexpr int kRepeat = 10;

struct DTypeEntry {
  common::DType act;
  common::DType weight;
  const char* label;
};

DTypeEntry g_specs[] = {
  {common::DType::kFp32,    common::DType::kFp32, "fp32"},
  {common::DType::kFp16,    common::DType::kFp16, "fp16"},
  {common::DType::kBf16,    common::DType::kBf16, "bf16"},
  {common::DType::kFp16,    common::DType::kFp32, "fp16_w32"},
  {common::DType::kBf16,    common::DType::kFp32, "bf16_w32"},
  {common::DType::kInt8,    common::DType::kFp32, "int8"},
  {common::DType::kFp8E4M3, common::DType::kFp32, "fp8_e4m3"},
  {common::DType::kFp8E5M2, common::DType::kFp32, "fp8_e5m2"},
};
constexpr int kNumSpecs = sizeof(g_specs) / sizeof(g_specs[0]);

bool IsDense(common::DType dt) {
  return dt == common::DType::kFp32 || dt == common::DType::kFp16 || dt == common::DType::kBf16;
}

bool IsQuantized(common::DType dt) {
  return dt == common::DType::kInt8 || dt == common::DType::kFp8E4M3 || dt == common::DType::kFp8E5M2;
}

bool IsDtypeSupported(common::DType dt) {
#if !(defined(CUDART_VERSION) && CUDART_VERSION >= 11080)
  if (dt == common::DType::kFp8E4M3 || dt == common::DType::kFp8E5M2) return false;
#endif
  return true;
}

struct GpuBuffers {
  void* input = nullptr;
  void* weight = nullptr;
  void* output = nullptr;
  float* input_scale = nullptr;
  float* output_scale = nullptr;
  common::DType act = common::DType::kFp32;
  bool quantized = false;
  int rows = 0;

  void Alloc(int rs, int cs, common::DType act_dt, common::DType wdt) {
    act = act_dt;
    quantized = IsQuantized(act_dt);
    rows = rs;
    const size_t n = static_cast<size_t>(rs) * cs;
    const size_t act_bytes = n * common::DTypeBytes(act_dt);
    const size_t w_bytes = static_cast<size_t>(cs) * common::DTypeBytes(wdt);
    CHECK_CUDA(cudaMalloc(&input, act_bytes));
    CHECK_CUDA(cudaMalloc(&weight, w_bytes));
    CHECK_CUDA(cudaMalloc(&output, act_bytes));
    if (quantized) {
      CHECK_CUDA(cudaMalloc(&input_scale, rs * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&output_scale, rs * sizeof(float)));
    }
  }

  ~GpuBuffers() {
    if (input) cudaFree(input);
    if (weight) cudaFree(weight);
    if (output) cudaFree(output);
    if (input_scale) cudaFree(input_scale);
    if (output_scale) cudaFree(output_scale);
  }
};

template <typename T>
void UploadDense(const std::vector<float>& x_f, const std::vector<float>& w_f,
                 void* d_x, void* d_w, int sz, int w_sz,
                 const std::vector<T>& x_dense, const std::vector<T>& w_dense) {
  CHECK_CUDA(cudaMemcpy(d_x, x_dense.data(), sz * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_w, w_dense.data(), w_sz * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void DownloadDense(void* d_out, std::vector<T>& out_dense, int sz) {
  out_dense.resize(sz);
  CHECK_CUDA(cudaMemcpy(out_dense.data(), d_out, sz * sizeof(T), cudaMemcpyDeviceToHost));
}

}  // namespace

int main() {
  std::cout << "=== RMSNorm Dtype Benchmark (rmsnorm::RmsNormRun) ===\n\n";

  int cuda_dev = 0;
  cudaGetDevice(&cuda_dev);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, cuda_dev);
  std::cout << "GPU: " << prop.name << "  (CC " << prop.major << "." << prop.minor << ")\n\n";

  const std::string results_dir = common::EnsureResultsDir();
  std::ofstream ofs(results_dir + "/rmsnorm_dtype_results.csv");
  ofs << "shape,rows,cols,dtype,weight_dtype,cpu_ms,gpu_ms,bandwidth_gbs,max_abs_diff,check\n";

  const int col_w = 12;
  const int col_shape = 37;
  std::cout << std::left
            << std::setw(col_shape) << "shape"
            << std::setw(8) << "rows"
            << std::setw(8) << "cols"
            << std::setw(col_w) << "dtype"
            << std::setw(col_w) << "CPU ms"
            << std::setw(col_w) << "GPU ms"
            << std::setw(10) << "GB/s"
            << std::setw(9) << "Check" << "\n";
  std::cout << std::string(115, '-') << "\n";

  const std::vector<std::tuple<int, int, const char*>> cases = {
      // tiny boundary shapes
      {1,      1,     "[tiny]  rows1_cols1"},
      {1,      4096,  "[tiny]  rows1_cols4K"},
      {4096,   1,     "[tiny]  rows4K_cols1"},
      {4,      4096,  "[tiny]  rows4_cols4K"},
      // small shapes
      {128,    64,    "[sm]    rows128_cols64"},
      {256,    64,    "[sm]    rows256_cols64"},
      {128,    256,   "[sm]    rows128_cols256"},
      // medium shapes (LLM hidden dimension range)
      {512,    512,   "[md]    rows512_cols512"},
      {1024,   512,   "[md]    rows1K_cols512"},
      {512,    1024,  "[md]    rows512_cols1K"},
      {1024,   1024,  "[md]    rows1K_cols1K"},
      {2048,   1024,  "[md]    rows2K_cols1K"},
      // large shapes
      {4096,   2048,  "[lg]    rows4K_cols2K"},
      {4096,   4096,  "[lg]    rows4K_cols4K"},
      {8192,   2048,  "[lg]    rows8K_cols2K"},
      {8192,   4096,  "[lg]    rows8K_cols4K"},
      {16384,  2048,  "[lg]    rows16K_cols2K"},
      {16384,  4096,  "[lg]    rows16K_cols4K"},
      // tiny cols (weight is small)
      {8192,   128,   "[tiny]  rows8K_cols128"},
      {16384,  128,   "[tiny]  rows16K_cols128"},
      {32768,  64,    "[tiny]  rows32K_cols64"},
      // unaligned
      {100,    100,   "[ua]    rows100_cols100"},
      {1023,   1011,  "[ua]    rows1023_cols1011"},
  };

  for (const auto& c : cases) {
    const int rows = std::get<0>(c);
    const int cols = std::get<1>(c);
    const char* shape_label = std::get<2>(c);
    const size_t n = static_cast<size_t>(rows) * cols;

    std::vector<float> w_fp32(cols);
    for (int ci = 0; ci < cols; ++ci)
      w_fp32[ci] = 0.9f + 0.001f * static_cast<float>(ci);

    std::vector<float> x_fp32(n);
    for (int r = 0; r < rows; ++r)
      for (int ci = 0; ci < cols; ++ci)
        x_fp32[static_cast<size_t>(r) * cols + ci] =
            0.3f * (static_cast<float>(r + 1) * 0.01f + static_cast<float>(ci + 1) * 0.001f);

    const bool do_cpu = (rows * cols <= 16 * 1024 * 1024);
    double cpu_ms = 0;
    std::vector<float> y_ref(n);
    if (do_cpu) {
      rmsnorm::RmsNormParams ref_p;
      ref_p.rows = rows; ref_p.cols = cols;
      ref_p.act_dtype = common::DType::kFp32;
      ref_p.weight_dtype = common::DType::kFp32;
      ref_p.input = x_fp32.data();
      ref_p.weight = w_fp32.data();
      ref_p.output = y_ref.data();
      const auto t0 = std::chrono::high_resolution_clock::now();
      if (!rmsnorm::RmsNormReferenceHost(ref_p).ok()) {
        std::cerr << "CPU ref failed\n";
        continue;
      }
      const auto t1 = std::chrono::high_resolution_clock::now();
      cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    for (int si = 0; si < kNumSpecs; ++si) {
      const common::DType act_dt = g_specs[si].act;
      const common::DType wdt = g_specs[si].weight;
      const char* dlabel = g_specs[si].label;

      if (!IsDtypeSupported(act_dt)) {
        std::cout << std::setw(col_shape) << shape_label
                  << std::setw(8) << rows << std::setw(8) << cols
                  << std::setw(col_w) << dlabel
                  << std::setw(col_w) << (si == 0 && do_cpu ? cpu_ms : 0.0)
                  << std::setw(col_w) << "SKIP"
                  << std::setw(10) << "-"
                  << std::setw(9) << "SKIP" << "\n";
        continue;
      }

      const size_t act_bytes = n * common::DTypeBytes(act_dt);

      std::vector<float> y_ref_quant;
      std::vector<float> y_scale_ref;
      const float* ref_scale_ptr = nullptr;
      const void* ref_out_ptr = nullptr;

      if (IsDense(act_dt)) {
        ref_out_ptr = y_ref.data();
      } else {
        rmsnorm::RmsNormParams qref;
        qref.rows = rows; qref.cols = cols;
        qref.act_dtype = act_dt;
        qref.weight_dtype = common::DType::kFp32;
        qref.eps = 1e-5f;
        rmsnorm::ActivationDtype adt = rmsnorm::ActivationDtype::kInt8;
        if (act_dt == common::DType::kFp8E4M3) adt = rmsnorm::ActivationDtype::kFp8E4M3;
        else if (act_dt == common::DType::kFp8E5M2) adt = rmsnorm::ActivationDtype::kFp8E5M2;
        std::vector<uint8_t> x_q_storage, y_q_storage;
        std::vector<float> x_scale(rows), y_scale(rows);
        rmsnorm::QuantizeActivationHost(adt, x_fp32, rows, cols,
                                        x_q_storage, x_scale);
        y_q_storage.assign(x_q_storage.size(), 0);
        qref.input = x_q_storage.data();
        qref.weight = w_fp32.data();
        qref.output = y_q_storage.data();
        qref.input_scale = x_scale.data();
        qref.output_scale = y_scale.data();
        if (!rmsnorm::RmsNormReferenceHost(qref).ok()) {
          std::cerr << "Quant CPU ref failed for " << dlabel << "\n";
          continue;
        }
        y_ref_quant = rmsnorm::DequantizeMatrixHost(adt, y_q_storage, y_scale, rows, cols);
        y_scale_ref = y_scale;
        ref_out_ptr = y_q_storage.data();
        ref_scale_ptr = y_scale.data();
      }

      GpuBuffers gpu;
      gpu.Alloc(rows, cols, act_dt, wdt);

      {
        const size_t w_bytes = static_cast<size_t>(cols) * common::DTypeBytes(wdt);
        if (wdt == common::DType::kFp32) {
          CHECK_CUDA(cudaMemcpy(gpu.weight, w_fp32.data(), w_bytes, cudaMemcpyHostToDevice));
        } else if (wdt == common::DType::kFp16) {
          std::vector<__half> w_h(cols);
          for (int ci = 0; ci < cols; ++ci) w_h[ci] = __float2half(w_fp32[ci]);
          CHECK_CUDA(cudaMemcpy(gpu.weight, w_h.data(), w_bytes, cudaMemcpyHostToDevice));
        } else {
          std::vector<__nv_bfloat16> w_bf(cols);
          for (int ci = 0; ci < cols; ++ci) w_bf[ci] = __float2bfloat16(w_fp32[ci]);
          CHECK_CUDA(cudaMemcpy(gpu.weight, w_bf.data(), w_bytes, cudaMemcpyHostToDevice));
        }
      }

      if (act_dt == common::DType::kFp32) {
        CHECK_CUDA(cudaMemcpy(gpu.input, x_fp32.data(), act_bytes, cudaMemcpyHostToDevice));
      } else if (act_dt == common::DType::kFp16) {
        std::vector<__half> x_h(n);
        for (size_t i = 0; i < n; ++i) x_h[i] = __float2half(x_fp32[i]);
        CHECK_CUDA(cudaMemcpy(gpu.input, x_h.data(), act_bytes, cudaMemcpyHostToDevice));
      } else if (act_dt == common::DType::kBf16) {
        std::vector<__nv_bfloat16> x_bf(n);
        for (size_t i = 0; i < n; ++i) x_bf[i] = __float2bfloat16(x_fp32[i]);
        CHECK_CUDA(cudaMemcpy(gpu.input, x_bf.data(), act_bytes, cudaMemcpyHostToDevice));
      } else {
        std::vector<uint8_t> x_q;
        std::vector<float> x_scale(rows);
        rmsnorm::ActivationDtype adt = rmsnorm::ActivationDtype::kInt8;
        if (act_dt == common::DType::kFp8E4M3) adt = rmsnorm::ActivationDtype::kFp8E4M3;
        else if (act_dt == common::DType::kFp8E5M2) adt = rmsnorm::ActivationDtype::kFp8E5M2;
        rmsnorm::QuantizeActivationHost(adt, x_fp32, rows, cols, x_q, x_scale);
        CHECK_CUDA(cudaMemcpy(gpu.input, x_q.data(), act_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(gpu.input_scale, x_scale.data(), rows * sizeof(float),
                              cudaMemcpyHostToDevice));
      }

      rmsnorm::RmsNormParams p;
      p.rows = rows; p.cols = cols;
      p.act_dtype = act_dt; p.weight_dtype = wdt;
      p.eps = 1e-5f;
      p.input = gpu.input; p.weight = gpu.weight; p.output = gpu.output;
      p.input_scale = gpu.input_scale; p.output_scale = gpu.output_scale;
      p.impl = rmsnorm::ImplId::kAuto;

      auto st = rmsnorm::RmsNormRun(p, nullptr);
      if (!st.ok()) {
        std::cout << std::setw(col_shape) << shape_label
                  << std::setw(8) << rows << std::setw(8) << cols
                  << std::setw(col_w) << dlabel
                  << std::setw(col_w) << (si == 0 && do_cpu ? cpu_ms : 0.0)
                  << std::setw(col_w) << "FAIL"
                  << std::setw(10) << "-"
                  << std::setw(9) << "FAIL" << "\n";
        continue;
      }
      CHECK_CUDA(cudaDeviceSynchronize());

      for (int w = 0; w < kWarmup; ++w) {
        rmsnorm::RmsNormRun(p, nullptr);
      }
      CHECK_CUDA(cudaDeviceSynchronize());

      cudaEvent_t s, e;
      CHECK_CUDA(cudaEventCreate(&s));
      CHECK_CUDA(cudaEventCreate(&e));
      std::vector<float> gpu_times;
      for (int rep = 0; rep < kRepeat; ++rep) {
        CHECK_CUDA(cudaEventRecord(s));
        auto run_st = rmsnorm::RmsNormRun(p, nullptr);
        if (!run_st.ok()) { std::cerr << "run fail: " << run_st.message << "\n"; break; }
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));
        CHECK_CUDA(cudaGetLastError());
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
        gpu_times.push_back(ms);
      }
      CHECK_CUDA(cudaEventDestroy(s));
      CHECK_CUDA(cudaEventDestroy(e));

      if (gpu_times.empty()) continue;
      std::sort(gpu_times.begin(), gpu_times.end());
      float gpu_ms = 0;
      if (gpu_times.size() > 2) {
        for (size_t ti = 1; ti + 1 < gpu_times.size(); ++ti) gpu_ms += gpu_times[ti];
        gpu_ms /= static_cast<float>(gpu_times.size() - 2);
      } else {
        for (float t : gpu_times) gpu_ms += t;
        gpu_ms /= static_cast<float>(gpu_times.size());
      }

      const double total_elems = static_cast<double>(rows) * cols + cols;
      const double gb_read = total_elems * static_cast<double>(common::DTypeBytes(act_dt));
      const double gb_written = static_cast<double>(rows) * cols * common::DTypeBytes(act_dt);
      const double gb_total = (gb_read + gb_written) / 1e9;
      const double bandwidth_gbs = (gpu_ms > 0) ? gb_total / (gpu_ms * 1e-3) : 0;

      std::vector<float> y_gpu(n);
      bool is_quant = IsQuantized(act_dt);
      if (is_quant) {
        std::vector<float> out_scales(rows);
        CHECK_CUDA(cudaMemcpy(out_scales.data(), gpu.output_scale, rows * sizeof(float),
                              cudaMemcpyDeviceToHost));
        const size_t act_b = static_cast<size_t>(rows) * cols * common::DTypeBytes(act_dt);
        std::vector<uint8_t> out_q(act_b);
        CHECK_CUDA(cudaMemcpy(out_q.data(), gpu.output, act_b, cudaMemcpyDeviceToHost));
        rmsnorm::ActivationDtype adt = rmsnorm::ActivationDtype::kInt8;
        if (act_dt == common::DType::kFp8E4M3) adt = rmsnorm::ActivationDtype::kFp8E4M3;
        else if (act_dt == common::DType::kFp8E5M2) adt = rmsnorm::ActivationDtype::kFp8E5M2;
        y_gpu = rmsnorm::DequantizeMatrixHost(adt, out_q, out_scales, rows, cols);
      } else if (act_dt == common::DType::kFp32) {
        CHECK_CUDA(cudaMemcpy(y_gpu.data(), gpu.output, n * sizeof(float), cudaMemcpyDeviceToHost));
      } else if (act_dt == common::DType::kFp16) {
        std::vector<__half> tmp(n);
        CHECK_CUDA(cudaMemcpy(tmp.data(), gpu.output, n * sizeof(__half), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) y_gpu[i] = __half2float(tmp[i]);
      } else {
        std::vector<__nv_bfloat16> tmp(n);
        CHECK_CUDA(cudaMemcpy(tmp.data(), gpu.output, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) y_gpu[i] = __bfloat162float(tmp[i]);
      }

      double max_diff = 0;
      bool check_ok = true;
      if (do_cpu) {
        const std::vector<float>& ref_vec = is_quant ? y_ref_quant : y_ref;
        max_diff = common::MaxAbsDiff(ref_vec, y_gpu);
        double tol = 1e-3;
        if (act_dt == common::DType::kFp16 || act_dt == common::DType::kBf16) tol = 0.05;
        else if (act_dt == common::DType::kInt8) tol = 5.0;
        else if (act_dt == common::DType::kFp8E4M3) tol = 1.0;
        else if (act_dt == common::DType::kFp8E5M2) tol = 2.0;
        check_ok = (max_diff < tol);
      }

      std::cout << std::left
                << std::setw(col_shape) << shape_label
                << std::setw(8) << rows << std::setw(8) << cols
                << std::setw(col_w) << dlabel
                << std::fixed << std::setprecision(3)
                << std::setw(col_w) << (si == 0 && do_cpu ? cpu_ms : 0.0)
                << std::setw(col_w) << gpu_ms
                << std::setw(10) << std::setprecision(1) << bandwidth_gbs
                << std::setw(9) << (check_ok ? "PASS" : "FAIL") << "\n";

      ofs << shape_label << "," << rows << "," << cols << ","
          << dlabel << "," << (wdt == common::DType::kFp32 ? "fp32" : dlabel) << ","
          << (si == 0 && do_cpu ? cpu_ms : 0.0) << "," << gpu_ms << ","
          << bandwidth_gbs << "," << max_diff << "," << (check_ok ? "PASS" : "FAIL") << "\n";
    }

    std::cout << std::string(115, '-') << "\n";
  }

  std::cout << "\nResults saved to " << results_dir << "/rmsnorm_dtype_results.csv\n";
  return 0;
}
