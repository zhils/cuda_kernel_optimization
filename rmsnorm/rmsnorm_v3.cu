// RMSNorm V3: weight 缓存在共享内存中，使用 warp shuffle 归约平方和，
// 通过 --dtype 参数支持 fp32、fp16、bf16 等多种数据类型。

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <vector>

#include "../common/include/common/benchmark.h"
#include "../common/include/common/cuda_utils.h"
#include "rmsnorm/rmsnorm_dtype.h"
#include "rmsnorm/rmsnorm_quant.h"
// CPU 参考实现定义见下方

static void RMSNormCPU(const float* x, const float* weight, float* y, int rows, int cols, float eps) {
  for (int i = 0; i < rows; ++i) {
    double ss = 0;
    for (int j = 0; j < cols; ++j) ss += (double)x[i * cols + j] * x[i * cols + j];
    double rms = sqrt(ss / cols + (double)eps);
    for (int j = 0; j < cols; ++j) y[i * cols + j] = x[i * cols + j] / (float)rms * weight[j];
  }
}

#include "rmsnorm/rmsnorm_v3_dtype.cuh"
#include "test_utils.h"

#define EPS 1e-5f

namespace {

template <typename ActT, typename WeightT>
std::vector<ActT> ConvertFromFloat(const std::vector<float>& src) {
  std::vector<ActT> out(src.size());
  if constexpr (std::is_same_v<ActT, float>) {
    out = src;
  } else if constexpr (std::is_same_v<ActT, __half>) {
    for (size_t i = 0; i < src.size(); ++i) out[i] = __float2half(src[i]);
  } else {
    for (size_t i = 0; i < src.size(); ++i) out[i] = __float2bfloat16(src[i]);
  }
  return out;
}

template <typename ActT>
std::vector<float> ConvertToFloat(const std::vector<ActT>& src) {
  std::vector<float> out(src.size());
  if constexpr (std::is_same_v<ActT, float>) {
    out = src;
  } else if constexpr (std::is_same_v<ActT, __half>) {
    for (size_t i = 0; i < src.size(); ++i) out[i] = __half2float(src[i]);
  } else {
    for (size_t i = 0; i < src.size(); ++i) out[i] = __bfloat162float(src[i]);
  }
  return out;
}

struct RunResult {
  float gpu_ms = 0.f;
  double bandwidth_gb_s = 0.0;
  double max_abs_diff = 0.0;
  bool ok = false;
};

template <typename ActT, typename WeightT, typename SetupFn, typename KernelLaunch>
RunResult BenchmarkCase(const std::vector<float>& x_fp32, const std::vector<float>& w_fp32,
                        int rows, int cols, rmsnorm::WeightDtype wdt, float tol,
                        SetupFn setup, KernelLaunch launch) {
  const int n = rows * cols;
  const auto x_host = ConvertFromFloat<ActT, WeightT>(x_fp32);
  const auto w_host = ConvertFromFloat<WeightT, WeightT>(w_fp32);
  const auto x_rounded = ConvertToFloat(x_host);
  const auto w_rounded = ConvertToFloat(w_host);

  std::vector<float> y_cpu_fp32(n);
  RMSNormCPU(x_rounded.data(), w_rounded.data(), y_cpu_fp32.data(), rows, cols, EPS);
  const auto y_cpu_act = ConvertFromFloat<ActT, WeightT>(y_cpu_fp32);
  const auto y_cpu_check = ConvertToFloat(y_cpu_act);

  ActT *dx = nullptr, *dy = nullptr;
  WeightT* dw = nullptr;
  CHECK_CUDA(cudaMalloc(&dx, n * sizeof(ActT)));
  CHECK_CUDA(cudaMalloc(&dy, n * sizeof(ActT)));
  CHECK_CUDA(cudaMalloc(&dw, cols * sizeof(WeightT)));
  CHECK_CUDA(cudaMemcpy(dx, x_host.data(), n * sizeof(ActT), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dw, w_host.data(), cols * sizeof(WeightT), cudaMemcpyHostToDevice));

  const size_t smem_size = static_cast<size_t>(cols) * sizeof(float);
  const dim3 block(rmsnorm::kV3BlockSize);
  const dim3 grid((rows + rmsnorm::kV3WarpsPerBlock - 1) / rmsnorm::kV3WarpsPerBlock);

  setup(smem_size);
  launch(dx, dy, dw, wdt, rows, cols, grid, block, smem_size);
  CHECK_CUDA(cudaDeviceSynchronize());

  constexpr int kRepeat = 10;
  cudaEvent_t s, e;
  CHECK_CUDA(cudaEventCreate(&s));
  CHECK_CUDA(cudaEventCreate(&e));
  CHECK_CUDA(cudaEventRecord(s));
  for (int rep = 0; rep < kRepeat; ++rep) {
    launch(dx, dy, dw, wdt, rows, cols, grid, block, smem_size);
  }
  CHECK_CUDA(cudaEventRecord(e));
  CHECK_CUDA(cudaEventSynchronize(e));
  float gpu_ms_total = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_total, s, e));

  std::vector<ActT> y_gpu(n);
  CHECK_CUDA(cudaMemcpy(y_gpu.data(), dy, n * sizeof(ActT), cudaMemcpyDeviceToHost));

  const auto y_gpu_fp32 = ConvertToFloat(y_gpu);
  RunResult result;
  result.gpu_ms = gpu_ms_total / static_cast<float>(kRepeat);
  result.max_abs_diff = common::MaxAbsDiff(y_cpu_check, y_gpu_fp32);
  result.ok = common::CheckEqual(y_cpu_check, y_gpu_fp32, tol);
  result.bandwidth_gb_s =
      static_cast<double>(n) * static_cast<double>(sizeof(ActT)) * 2.0 /
      (static_cast<double>(result.gpu_ms) * 1e6);

  CHECK_CUDA(cudaEventDestroy(s));
  CHECK_CUDA(cudaEventDestroy(e));
  CHECK_CUDA(cudaFree(dx));
  CHECK_CUDA(cudaFree(dy));
  CHECK_CUDA(cudaFree(dw));
  return result;
}

template <typename WeightT, typename SetupFn, typename KernelLaunch>
RunResult BenchmarkQuantCase(rmsnorm::ActivationDtype act_dt, const std::vector<float>& x_fp32,
                             const std::vector<float>& w_fp32, int rows, int cols,
                             rmsnorm::WeightDtype wdt, float tol, SetupFn setup,
                             KernelLaunch launch) {
  const int n = rows * cols;
  std::vector<uint8_t> x_q, y_cpu_q, y_gpu_q;
  std::vector<float> x_scale, y_cpu_scale, y_gpu_scale;
  rmsnorm::QuantizeActivationHost(act_dt, x_fp32, rows, cols, x_q, x_scale);
  rmsnorm::RMSNormQuantizedCPU(act_dt, x_q, x_scale, w_fp32, y_cpu_q, y_cpu_scale, rows, cols,
                               EPS);

  const auto w_host = ConvertFromFloat<WeightT, WeightT>(w_fp32);

  uint8_t *dx = nullptr, *dy = nullptr;
  float *dx_scale = nullptr, *dy_scale = nullptr;
  WeightT* dw = nullptr;
  CHECK_CUDA(cudaMalloc(&dx, n));
  CHECK_CUDA(cudaMalloc(&dy, n));
  CHECK_CUDA(cudaMalloc(&dx_scale, rows * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dy_scale, rows * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dw, cols * sizeof(WeightT)));
  CHECK_CUDA(cudaMemcpy(dx, x_q.data(), n, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dx_scale, x_scale.data(), rows * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dw, w_host.data(), cols * sizeof(WeightT), cudaMemcpyHostToDevice));

  const size_t smem_size = static_cast<size_t>(cols) * sizeof(float);
  const dim3 block(rmsnorm::kV3BlockSize);
  const dim3 grid((rows + rmsnorm::kV3WarpsPerBlock - 1) / rmsnorm::kV3WarpsPerBlock);
  const float quant_max = rmsnorm::QuantMax(act_dt);

  setup(smem_size);
  launch(dx, dy, dx_scale, dy_scale, dw, wdt, rows, cols, grid, block, smem_size, quant_max);
  CHECK_CUDA(cudaDeviceSynchronize());

  constexpr int kRepeat = 10;
  cudaEvent_t s, e;
  CHECK_CUDA(cudaEventCreate(&s));
  CHECK_CUDA(cudaEventCreate(&e));
  CHECK_CUDA(cudaEventRecord(s));
  for (int rep = 0; rep < kRepeat; ++rep) {
    launch(dx, dy, dx_scale, dy_scale, dw, wdt, rows, cols, grid, block, smem_size, quant_max);
  }
  CHECK_CUDA(cudaEventRecord(e));
  CHECK_CUDA(cudaEventSynchronize(e));
  float gpu_ms_total = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_total, s, e));

  y_gpu_q.resize(n);
  y_gpu_scale.resize(rows);
  CHECK_CUDA(cudaMemcpy(y_gpu_q.data(), dy, n, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(y_gpu_scale.data(), dy_scale, rows * sizeof(float),
                        cudaMemcpyDeviceToHost));

  const auto y_cpu_f =
      rmsnorm::DequantizeMatrixHost(act_dt, y_cpu_q, y_cpu_scale, rows, cols);
  const auto y_gpu_f =
      rmsnorm::DequantizeMatrixHost(act_dt, y_gpu_q, y_gpu_scale, rows, cols);

  RunResult result;
  result.gpu_ms = gpu_ms_total / static_cast<float>(kRepeat);
  result.max_abs_diff = common::MaxAbsDiff(y_cpu_f, y_gpu_f);
  result.ok = common::CheckEqual(y_cpu_f, y_gpu_f, tol);
  result.bandwidth_gb_s = static_cast<double>(n) * 2.0 /
                          (static_cast<double>(result.gpu_ms) * 1e6);

  CHECK_CUDA(cudaEventDestroy(s));
  CHECK_CUDA(cudaEventDestroy(e));
  CHECK_CUDA(cudaFree(dx));
  CHECK_CUDA(cudaFree(dy));
  CHECK_CUDA(cudaFree(dx_scale));
  CHECK_CUDA(cudaFree(dy_scale));
  CHECK_CUDA(cudaFree(dw));
  return result;
}

template <typename WeightT, typename SetupFn, typename KernelLaunch>
RunResult DispatchQuant(rmsnorm::ActivationDtype act_dt, const std::vector<float>& x,
                        const std::vector<float>& w, int rows, int cols, rmsnorm::WeightDtype wdt,
                        float tol, SetupFn setup, KernelLaunch launch) {
  return BenchmarkQuantCase<WeightT>(act_dt, x, w, rows, cols, wdt, tol, setup, launch);
}

RunResult DispatchBenchmark(const rmsnorm::LaunchConfig& cfg, const std::vector<float>& x,
                            const std::vector<float>& w, int rows, int cols) {
  using WD = rmsnorm::WeightDtype;
  using AD = rmsnorm::ActivationDtype;
  const WD wdt = cfg.weight_dtype;

  if (cfg.act_dtype == AD::kFp32) {
    const auto setup = [](size_t smem) {
      CHECK_CUDA(cudaFuncSetAttribute(rmsnorm::RMSNormV3KernelFp32,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem)));
    };
    const auto launch = [](float* dx, float* dy, void* dw, WD wdt_in, int r, int c, dim3 grid,
                           dim3 block, size_t smem) {
      rmsnorm::RMSNormV3KernelFp32<<<grid, block, smem>>>(dx, dy, dw, wdt_in, r, c, EPS);
    };
    if (wdt == WD::kFp32) {
      return BenchmarkCase<float, float>(x, w, rows, cols, wdt, 1e-4f, setup, launch);
    }
    if (wdt == WD::kFp16) {
      return BenchmarkCase<float, __half>(x, w, rows, cols, wdt, 1e-4f, setup, launch);
    }
    return BenchmarkCase<float, __nv_bfloat16>(x, w, rows, cols, wdt, 1e-4f, setup, launch);
  }

  if (cfg.act_dtype == rmsnorm::ActivationDtype::kFp16) {
    const auto setup = [](size_t smem) {
      CHECK_CUDA(cudaFuncSetAttribute(rmsnorm::RMSNormV3KernelFp16,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem)));
    };
    const auto launch = [](__half* dx, __half* dy, void* dw, WD wdt_in, int r, int c, dim3 grid,
                           dim3 block, size_t smem) {
      rmsnorm::RMSNormV3KernelFp16<<<grid, block, smem>>>(dx, dy, dw, wdt_in, r, c, EPS);
    };
    if (wdt == WD::kFp32) {
      return BenchmarkCase<__half, float>(x, w, rows, cols, wdt, 1e-2f, setup, launch);
    }
    if (wdt == WD::kFp16) {
      return BenchmarkCase<__half, __half>(x, w, rows, cols, wdt, 1e-2f, setup, launch);
    }
    return BenchmarkCase<__half, __nv_bfloat16>(x, w, rows, cols, wdt, 1e-2f, setup, launch);
  }

  if (cfg.act_dtype == AD::kBf16) {
  const auto setup = [](size_t smem) {
    CHECK_CUDA(cudaFuncSetAttribute(rmsnorm::RMSNormV3KernelBf16,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(smem)));
  };
  const auto launch = [](__nv_bfloat16* dx, __nv_bfloat16* dy, void* dw, WD wdt_in, int r, int c,
                         dim3 grid, dim3 block, size_t smem) {
    rmsnorm::RMSNormV3KernelBf16<<<grid, block, smem>>>(dx, dy, dw, wdt_in, r, c, EPS);
  };
  constexpr float kBf16Tol = 2e-2f;
  if (wdt == WD::kFp32) {
    return BenchmarkCase<__nv_bfloat16, float>(x, w, rows, cols, wdt, kBf16Tol, setup, launch);
  }
  if (wdt == WD::kFp16) {
    return BenchmarkCase<__nv_bfloat16, __half>(x, w, rows, cols, wdt, kBf16Tol, setup, launch);
  }
  return BenchmarkCase<__nv_bfloat16, __nv_bfloat16>(x, w, rows, cols, wdt, kBf16Tol, setup,
                                                     launch);
  }

  if (cfg.act_dtype == AD::kInt8) {
    const auto setup = [](size_t smem) {
      CHECK_CUDA(cudaFuncSetAttribute(rmsnorm::RMSNormV3KernelInt8,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem)));
    };
    const auto launch = [](uint8_t* dx, uint8_t* dy, float* dx_scale, float* dy_scale, void* dw,
                           WD wdt_in, int r, int c, dim3 grid, dim3 block, size_t smem,
                           float quant_max) {
      rmsnorm::RMSNormV3KernelInt8<<<grid, block, smem>>>(
          reinterpret_cast<const int8_t*>(dx), reinterpret_cast<int8_t*>(dy), dx_scale, dy_scale,
          dw, wdt_in, r, c, EPS, quant_max);
    };
    constexpr float kTol = 0.5f;
    if (wdt == WD::kFp32) {
      return DispatchQuant<float>(AD::kInt8, x, w, rows, cols, wdt, kTol, setup, launch);
    }
    if (wdt == WD::kFp16) {
      return DispatchQuant<__half>(AD::kInt8, x, w, rows, cols, wdt, kTol, setup, launch);
    }
    return DispatchQuant<__nv_bfloat16>(AD::kInt8, x, w, rows, cols, wdt, kTol, setup, launch);
  }

  if (cfg.act_dtype == AD::kFp8E4M3) {
    const auto setup = [](size_t smem) {
      CHECK_CUDA(cudaFuncSetAttribute(rmsnorm::RMSNormV3KernelFp8E4M3,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem)));
    };
    const auto launch = [](uint8_t* dx, uint8_t* dy, float* dx_scale, float* dy_scale, void* dw,
                           WD wdt_in, int r, int c, dim3 grid, dim3 block, size_t smem,
                           float quant_max) {
      rmsnorm::RMSNormV3KernelFp8E4M3<<<grid, block, smem>>>(
          reinterpret_cast<const __nv_fp8_e4m3*>(dx),
          reinterpret_cast<__nv_fp8_e4m3*>(dy), dx_scale, dy_scale, dw, wdt_in, r, c, EPS,
          quant_max);
    };
    constexpr float kTol = 1.0f;
    if (wdt == WD::kFp32) {
      return DispatchQuant<float>(AD::kFp8E4M3, x, w, rows, cols, wdt, kTol, setup, launch);
    }
    if (wdt == WD::kFp16) {
      return DispatchQuant<__half>(AD::kFp8E4M3, x, w, rows, cols, wdt, kTol, setup, launch);
    }
    return DispatchQuant<__nv_bfloat16>(AD::kFp8E4M3, x, w, rows, cols, wdt, kTol, setup, launch);
  }

  if (cfg.act_dtype == AD::kFp8E5M2) {
    const auto setup = [](size_t smem) {
      CHECK_CUDA(cudaFuncSetAttribute(rmsnorm::RMSNormV3KernelFp8E5M2,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem)));
    };
    const auto launch = [](uint8_t* dx, uint8_t* dy, float* dx_scale, float* dy_scale, void* dw,
                           WD wdt_in, int r, int c, dim3 grid, dim3 block, size_t smem,
                           float quant_max) {
      rmsnorm::RMSNormV3KernelFp8E5M2<<<grid, block, smem>>>(
          reinterpret_cast<const __nv_fp8_e5m2*>(dx),
          reinterpret_cast<__nv_fp8_e5m2*>(dy), dx_scale, dy_scale, dw, wdt_in, r, c, EPS,
          quant_max);
    };
    constexpr float kTol = 2.0f;
    if (wdt == WD::kFp32) {
      return DispatchQuant<float>(AD::kFp8E5M2, x, w, rows, cols, wdt, kTol, setup, launch);
    }
    if (wdt == WD::kFp16) {
      return DispatchQuant<__half>(AD::kFp8E5M2, x, w, rows, cols, wdt, kTol, setup, launch);
    }
    return DispatchQuant<__nv_bfloat16>(AD::kFp8E5M2, x, w, rows, cols, wdt, kTol, setup, launch);
  }

  return {};
}

}  // namespace

int main(int argc, char** argv) {
  // 解析配置并设置默认权重类型
  rmsnorm::LaunchConfig cfg = rmsnorm::ParseArgs(argc, argv);
  bool weight_explicit = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--weight-dtype") == 0) weight_explicit = true;
  }
  if (!weight_explicit) {
    switch (cfg.act_dtype) {
      case rmsnorm::ActivationDtype::kFp32:
        cfg.weight_dtype = rmsnorm::WeightDtype::kFp32;
        break;
      case rmsnorm::ActivationDtype::kFp16:
        cfg.weight_dtype = rmsnorm::WeightDtype::kFp16;
        break;
      case rmsnorm::ActivationDtype::kBf16:
        cfg.weight_dtype = rmsnorm::WeightDtype::kBf16;
        break;
      case rmsnorm::ActivationDtype::kInt8:
      case rmsnorm::ActivationDtype::kFp8E4M3:
      case rmsnorm::ActivationDtype::kFp8E5M2:
        cfg.weight_dtype = rmsnorm::WeightDtype::kFp32;
        break;
      default:
        break;
    }
  }

  // 验证配置并准备结果文件
  rmsnorm::ValidateConfig(cfg);

  constexpr int kTestCases = 5;
  const std::string results_dir = common::EnsureResultsDir();
  std::ofstream ofs(results_dir + "/rmsnorm_v3_results.csv");
  ofs << "id,rows,cols,act_dtype,weight_dtype,gpu_ms,bandwidth_gb_s,max_abs_diff,check\n";

  const std::vector<std::pair<int, int>> test_sizes = {
      {128, 128}, {256, 256}, {512, 512}, {1024, 1024}, {4096, 4096}};

  std::cout << "=== RMSNorm V3 (dtype=" << rmsnorm::ActivationDtypeName(cfg.act_dtype)
            << ", weight=" << rmsnorm::WeightDtypeName(cfg.weight_dtype) << ") ===\n";

  // 逐用例测试
  for (int i = 0; i < kTestCases; ++i) {
    const int rows = test_sizes[i].first;
    const int cols = test_sizes[i].second;
    const std::vector<float> x = rmsnorm::RandomMatrix(rows, cols, 2026 + i);
    const std::vector<float> w = rmsnorm::RandomWeight(cols, 2026 + i + 100);
    const RunResult result = DispatchBenchmark(cfg, x, w, rows, cols);

    std::cout << rows << "x" << cols
              << " | BlockSize=" << rmsnorm::kV3BlockSize
              << " | " << std::fixed << std::setprecision(4) << result.gpu_ms << " ms"
              << " | " << std::setprecision(1) << result.bandwidth_gb_s << " GB/s"
              << " | " << (result.ok ? "PASS" : "FAIL") << "\n";

    ofs << i << "," << rows << "," << cols << ","
        << rmsnorm::ActivationDtypeName(cfg.act_dtype) << ","
        << rmsnorm::WeightDtypeName(cfg.weight_dtype) << ","
        << result.gpu_ms << "," << result.bandwidth_gb_s << ","
        << result.max_abs_diff << "," << (result.ok ? "PASS" : "FAIL") << "\n";
  }

  std::cout << "\nResults saved to " << results_dir << "/rmsnorm_v3_results.csv\n";
  return 0;
}
