#include <cuda_runtime.h>

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
#include "rmsnorm/rmsnorm_ref_cpu.h"

namespace {

constexpr int kWarmup = 3;
constexpr int kRepeat = 10;

struct ImplEntry {
  const char* label;
  rmsnorm::ImplId impl;
};

ImplEntry g_impls[] = {
  {"V3",      rmsnorm::ImplId::kV3},
  {"V2",      rmsnorm::ImplId::kV2},
  {"V1",      rmsnorm::ImplId::kV1},
  {"V0",      rmsnorm::ImplId::kV0},
  {"CubRef",  rmsnorm::ImplId::kCubRef},
};

constexpr int kNumImpls = sizeof(g_impls) / sizeof(g_impls[0]);

struct ImplResult {
  float gpu_ms = 0;
  double bandwidth_gbs = 0;
  bool ok = false;
  bool skipped = false;
  double max_abs_diff = 0;
  bool check_pass = true;
};

ImplResult RunOneImpl(rmsnorm::RmsNormParams p_template, int rows, int cols,
                       const ImplEntry& ie, void* d_x, void* d_w, void* d_y,
                       const float* y_ref, size_t n) {
  ImplResult r;

  rmsnorm::RmsNormParams p = p_template;
  p.impl = ie.impl;

  for (int w = 0; w < kWarmup; ++w) {
    auto st = rmsnorm::RmsNormRun(p, nullptr);
    if (!st.ok()) {
      r.skipped = true;
      return r;
    }
  }
  {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      cudaGetLastError();
      r.skipped = true;
      return r;
    }
  }

  cudaEvent_t s, e;
  CHECK_CUDA(cudaEventCreate(&s));
  CHECK_CUDA(cudaEventCreate(&e));
  std::vector<float> gpu_times;
  for (int rep = 0; rep < kRepeat; ++rep) {
    CHECK_CUDA(cudaEventRecord(s));
    auto st = rmsnorm::RmsNormRun(p, nullptr);
    if (!st.ok()) {
      r.skipped = true;
      break;
    }
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    {
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        r.skipped = true;
        break;
      }
    }
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
    gpu_times.push_back(ms);
  }
  CHECK_CUDA(cudaEventDestroy(s));
  CHECK_CUDA(cudaEventDestroy(e));

  if (r.skipped || gpu_times.empty()) {
    r.skipped = true;
    return r;
  }

  std::sort(gpu_times.begin(), gpu_times.end());
  if (gpu_times.size() > 2) {
    for (size_t ti = 1; ti + 1 < gpu_times.size(); ++ti) r.gpu_ms += gpu_times[ti];
    r.gpu_ms /= static_cast<float>(gpu_times.size() - 2);
  } else {
    for (float t : gpu_times) r.gpu_ms += t;
    r.gpu_ms /= static_cast<float>(gpu_times.size());
  }

  const double bytes = 2.0 * static_cast<double>(rows) * cols * sizeof(float);
  r.bandwidth_gbs = (r.gpu_ms > 0) ? bytes / (r.gpu_ms * 1e6) : 0;

  std::vector<float> y_gpu(n);
  CHECK_CUDA(cudaMemcpy(y_gpu.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));

  r.max_abs_diff = common::MaxAbsDiff(std::vector<float>(y_ref, y_ref + n), y_gpu);
  r.check_pass = (r.max_abs_diff < 1e-3f);

  r.ok = true;
  return r;
}

}  // namespace

int main() {
  std::cout << "=== RMSNorm Head-to-Head Implementation Comparison ===\n\n";
  std::cout << "GPU: ";
  {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    std::cout << prop.name << "  (CC " << prop.major << "." << prop.minor << ")\n";
  }
  std::cout << "Baseline: V3 (warp-level reduction, fp32 act/weight)\n\n";

  const std::vector<std::tuple<int, int, const char*>> cases = {
      {128,    64,    "[sm]    128x64"},
      {256,    64,    "[sm]    256x64"},
      {512,    512,   "[md]    512x512"},
      {1024,   512,   "[md]    1024x512"},
      {512,    1024,  "[md]    512x1024"},
      {1024,   1024,  "[md]    1024x1024"},
      {2048,   1024,  "[md]    2048x1024"},
      {4096,   2048,  "[lg]    4096x2048"},
      {4096,   4096,  "[lg]    4096x4096"},
      {8192,   2048,  "[lg]    8192x2048"},
      {8192,   4096,  "[lg]    8192x4096"},
      {16384,  2048,  "[lg]    16384x2048"},
      {8192,   128,   "[tiny]  8192x128"},
      {16384,  128,   "[tiny]  16384x128"},
      {32768,  64,    "[tiny]  32768x64"},
      {1,      1,     "[tiny]  1x1"},
      {1,      4096,  "[tiny]  1x4096"},
      {4096,   1,     "[tiny]  4096x1"},
      {4,      4096,  "[tiny]  4x4096"},
      {100,    100,   "[ua]    100x100"},
  };

  const int col_shape = 20;
  const int col_impl = 8;
  const int col_w = 12;
  const int col_sp = 16;
  std::cout << std::left
            << std::setw(col_shape) << "shape"
            << std::setw(8) << "rows"
            << std::setw(8) << "cols"
            << std::setw(col_impl) << "impl"
            << std::setw(col_w) << "GB/s"
            << std::setw(col_sp) << "vs V3 baseline"
            << std::setw(9) << "Check" << "\n";
  std::cout << std::string(94, '-') << "\n";

  const std::string results_dir = common::EnsureResultsDir();
  std::ofstream ofs(results_dir + "/rmsnorm_compare_results.csv");
  ofs << "shape,rows,cols,impl,gbs,speedup_vs_v3,baseline_gbs,check\n";

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  for (const auto& c : cases) {
    const int rows = std::get<0>(c);
    const int cols = std::get<1>(c);
    const char* shape_label = std::get<2>(c);
    const size_t n = static_cast<size_t>(rows) * cols;

    std::vector<float> x_fp32(n), w_fp32(cols);
    for (int r = 0; r < rows; ++r)
      for (int ci = 0; ci < cols; ++ci)
        x_fp32[static_cast<size_t>(r) * cols + ci] =
            0.3f * (static_cast<float>(r + 1) * 0.01f + static_cast<float>(ci + 1) * 0.001f);
    for (int ci = 0; ci < cols; ++ci)
      w_fp32[ci] = 0.9f + 0.001f * static_cast<float>(ci);

    rmsnorm::RmsNormParams ref_p;
    ref_p.rows = rows; ref_p.cols = cols;
    ref_p.act_dtype = common::DType::kFp32;
    ref_p.weight_dtype = common::DType::kFp32;
    ref_p.input = x_fp32.data();
    ref_p.weight = w_fp32.data();
    ref_p.output = nullptr;
    std::vector<float> y_ref(n);
    {
      rmsnorm::RmsNormParams ref_copy = ref_p;
      const int result_rows = ref_p.rows * ref_p.cols;
      std::vector<float> y_tmp(result_rows);
      ref_copy.output = y_tmp.data();
      auto st = rmsnorm::RmsNormReferenceHost(ref_copy);
      if (!st.ok()) { std::cerr << "Ref failed: " << st.message << "\n"; return 1; }
      y_ref = y_tmp;
    }

    float *d_x = nullptr, *d_w = nullptr, *d_y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, x_fp32.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, w_fp32.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    rmsnorm::RmsNormParams p_template;
    p_template.rows = rows;
    p_template.cols = cols;
    p_template.act_dtype = common::DType::kFp32;
    p_template.weight_dtype = common::DType::kFp32;
    p_template.eps = 1e-5f;
    p_template.input = d_x;
    p_template.weight = d_w;
    p_template.output = d_y;

    ImplResult results[kNumImpls];

    int baseline_idx = 0;
    for (int ii = 0; ii < kNumImpls; ++ii) {
      if (g_impls[ii].impl == rmsnorm::ImplId::kV3) { baseline_idx = ii; break; }
    }

    for (int ii = 0; ii < kNumImpls; ++ii) {
      results[ii] = RunOneImpl(p_template, rows, cols, g_impls[ii],
                                d_x, d_w, d_y, y_ref.data(), n);
    }

    const double baseline_gbs = (results[baseline_idx].ok && !results[baseline_idx].skipped)
                                    ? results[baseline_idx].bandwidth_gbs : 0;

    for (int ii = 0; ii < kNumImpls; ++ii) {
      const auto& ie = g_impls[ii];
      const auto& r = results[ii];

      std::cout << std::left << std::fixed
                << std::setw(col_shape) << shape_label
                << std::setw(8) << rows
                << std::setw(8) << cols
                << std::setw(col_impl) << ie.label;

      if (r.skipped) {
        std::cout << std::setw(col_w) << "-"
                  << std::setw(col_sp) << "-"
                  << std::setw(9) << "SKIP" << "\n";

        ofs << shape_label << "," << rows << "," << cols << ","
            << ie.label << ",0,0,0,SKIP\n";
        continue;
      }

      const double speedup = (baseline_gbs > 0) ? r.bandwidth_gbs / baseline_gbs : 0;

      std::cout << std::setprecision(1) << std::setw(col_w) << r.bandwidth_gbs
                << std::setw(col_sp);
      if (baseline_gbs > 0) {
        std::cout << std::setprecision(2) << speedup << "x";
      } else {
        std::cout << "-";
      }
      std::cout << std::setw(9) << (r.check_pass ? "PASS" : "FAIL") << "\n";

      ofs << shape_label << "," << rows << "," << cols << ","
          << ie.label << "," << r.bandwidth_gbs << "," << speedup << ","
          << baseline_gbs << "," << (r.check_pass ? "PASS" : "FAIL") << "\n";
    }

    std::cout << std::string(94, '-') << "\n";
    CHECK_CUDA(cudaFree(d_x)); CHECK_CUDA(cudaFree(d_w)); CHECK_CUDA(cudaFree(d_y));
  }

  std::cout << "\nResults saved to " << results_dir << "/rmsnorm_compare_results.csv\n";
  return 0;
}
