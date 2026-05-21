#include <cstdlib>
#include <iostream>
#include <string>

#include "gemm/gemm_tuner.h"

namespace {

const char* ImplName(gemm::ImplId id) {
  switch (id) {
    case gemm::ImplId::kAuto:        return "kAuto";
    case gemm::ImplId::kV0:          return "kV0";
    case gemm::ImplId::kV1:          return "kV1";
    case gemm::ImplId::kV2:          return "kV2";
    case gemm::ImplId::kV3:          return "kV3";
    case gemm::ImplId::kV4:          return "kV4";
    case gemm::ImplId::kFp16:        return "kFp16";
    case gemm::ImplId::kCublas:      return "kCublas";
    case gemm::ImplId::kCublasFp16:  return "kCublasFp16";
    case gemm::ImplId::kCublasBf16:  return "kCublasBf16";
    case gemm::ImplId::kCublasInt8:  return "kCublasInt8";
    case gemm::ImplId::kCublasFp8:   return "kCublasFp8";
  }
  return "unknown";
}

void PrintUsage() {
  std::cout << "GEMM Auto-Tuner – recommends optimal (dtype, impl) for given shape\n\n";
  std::cout << "Usage: gemm_tune M N K [strategy]\n\n";
  std::cout << "  M, N, K        GEMM dimensions: C(M,N) = A(M,K) * B(K,N)\n";
  std::cout << "  strategy        max_perf (default) | balanced | max_precision\n\n";
  std::cout << "Strategies:\n";
  std::cout << "  max_perf        Raw speed – may use int8/fp8 for maximum throughput\n";
  std::cout << "  balanced        Good speed + reasonable precision – prefers bf16/fp8_e4m3\n";
  std::cout << "  max_precision   FP32 only – guaranteed bit-exact correctness\n\n";
  std::cout << "Examples:\n";
  std::cout << "  gemm_tune 4096 4096 4096              # recommends fp8_e4m3\n";
  std::cout << "  gemm_tune 1024 1024 1024 balanced     # recommends fp8_e4m3 (balanced)\n";
  std::cout << "  gemm_tune 128 128 128                 # recommends V3 (small K, no TC)\n";
  std::cout << "  gemm_tune 100 100 100 max_precision   # fp32 cuBLAS\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    PrintUsage();
    return 1;
  }

  const int M = std::atoi(argv[1]);
  const int N = std::atoi(argv[2]);
  const int K = std::atoi(argv[3]);

  gemm::TuningStrategy strategy = gemm::TuningStrategy::kMaxPerformance;
  if (argc >= 5) {
    const std::string s(argv[4]);
    if (s == "balanced")
      strategy = gemm::TuningStrategy::kBalanced;
    else if (s == "max_precision")
      strategy = gemm::TuningStrategy::kMaxPrecision;
    else if (s != "max_perf") {
      std::cerr << "Unknown strategy: " << s << "\n";
      return 1;
    }
  }

  const auto r = gemm::TuneGemm(M, N, K, strategy);

  const double gflops_hint = 2.0 * M * N * K / 1e9;

  std::cout << "╔══════════════════════════════════════╗\n";
  std::cout << "║        GEMM Auto-Tuner Result        ║\n";
  std::cout << "╠══════════════════════════════════════╣\n";
  std::cout << "║ Shape   M=" << M << "  N=" << N << "  K=" << K << "\n";
  std::cout << "║ FLOPs   " << gflops_hint << " G\n";
  std::cout << "║ Strategy: " << gemm::StrategyName(strategy) << "\n";
  std::cout << "╠══════════════════════════════════════╣\n";
  std::cout << "║ dtype   " << common::DTypeName(r.dtype_a)
            << " → " << common::DTypeName(r.dtype_c) << "\n";
  std::cout << "║ impl    " << ImplName(r.impl) << "\n";
  std::cout << "║ reason  " << r.reason << "\n";
  std::cout << "╚══════════════════════════════════════╝\n";

  const double bw_factor = static_cast<double>(common::DTypeBytes(r.dtype_a)) / 4.0;
  std::cout << "\nBandwidth saving: "
            << static_cast<int>((1.0 - bw_factor) * 100) << "% vs fp32 "
            << "(A+B: " << common::DTypeBytes(r.dtype_a) << " bytes/elem)\n";

  return 0;
}
