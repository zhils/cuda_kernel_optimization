// FP8 (E4M3) GEMM performance via cuBLASLt Tensor Core path.
// Layout follows NVIDIA LtFp8Matmul sample: transA=T, transB=N (required on sm_89+ / sm_120).
// Reference: https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtFp8Matmul
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"

#define CHECK_CUBLASLT(call)                                                     \
  do {                                                                           \
    cublasStatus_t s__ = (call);                                                 \
    if (s__ != CUBLAS_STATUS_SUCCESS) {                                          \
      std::cerr << "cuBLASLt error: " << static_cast<int>(s__) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                    \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

static void GemmCpuFp32(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c) {
      float s = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        s += A[static_cast<size_t>(r) * K + kk] * B[static_cast<size_t>(kk) * N + c];
      }
      C[static_cast<size_t>(r) * N + c] = s;
    }
  }
}

// Column-major K×M: (i,j) -> i + j*ld, value = A_row[j*K+i]
static void FloatToFp8ColMajorKM(const std::vector<float>& A_row_mk, int M, int K,
                                 std::vector<__nv_fp8_e4m3>& out, int ld) {
  out.assign(static_cast<size_t>(ld) * M, __nv_fp8_e4m3(0.0f));
  for (int j = 0; j < M; ++j) {
    for (int i = 0; i < K; ++i) {
      float v = A_row_mk[static_cast<size_t>(j) * K + i];
      out[static_cast<size_t>(i) + static_cast<size_t>(j) * ld] = __nv_fp8_e4m3(v);
    }
  }
}

// Column-major K×N
static void FloatToFp8ColMajorKN(const std::vector<float>& B_row_kn, int K, int N,
                                   std::vector<__nv_fp8_e4m3>& out, int ld) {
  out.assign(static_cast<size_t>(ld) * N, __nv_fp8_e4m3(0.0f));
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < K; ++i) {
      float v = B_row_kn[static_cast<size_t>(i) * N + j];
      out[static_cast<size_t>(i) + static_cast<size_t>(j) * ld] = __nv_fp8_e4m3(v);
    }
  }
}

static void ColMajorMNToRowMajor(const float* col, int M, int N, int ldc, std::vector<float>& row) {
  row.resize(static_cast<size_t>(M) * N);
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c) {
      row[static_cast<size_t>(r) * N + c] = col[static_cast<size_t>(r) + static_cast<size_t>(c) * ldc];
    }
  }
}

int main() {
  constexpr int kWarmup = 3;
  constexpr int kRepeat = 10;
  constexpr int kMaxCpuVerifyDim = 1024;
  constexpr size_t kWorkspaceBytes = 32ull * 1024 * 1024;
  constexpr cublasOperation_t kTransA = CUBLAS_OP_T;
  constexpr cublasOperation_t kTransB = CUBLAS_OP_N;

  auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/gemm_fp8_cublaslt_results.csv");
  ofs << "id,group,M,N,K,gpu_ms,gflops,max_abs_diff,check,note\n";

  cublasLtHandle_t lt{};
  CHECK_CUBLASLT(cublasLtCreate(&lt));

  void* workspace = nullptr;
  CHECK_CUDA(cudaMalloc(&workspace, kWorkspaceBytes));

  std::cout << "cuBLASLt FP8 E4M3 matmul (TN layout), D=FP32\n";
  std::cout << "===========================================\n\n";

  for (size_t ci = 0; ci < cases.size(); ++ci) {
    const int M = cases[ci].rows;
    const int N = cases[ci].cols;
    const int K = M;
    const int m = M;
    const int n = N;
    const int kdim = K;

    std::vector<float> A_row(static_cast<size_t>(M) * K);
    std::vector<float> B_row(static_cast<size_t>(K) * N);
    std::vector<float> C_cpu(static_cast<size_t>(M) * N);
    common::InitMatrix(A_row, M, K);
    common::InitMatrix(B_row, K, N);

    const bool do_cpu_verify = (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim);
    if (do_cpu_verify) {
      GemmCpuFp32(A_row.data(), B_row.data(), C_cpu.data(), M, N, K);
    }

    const int ld_a = kdim;
    const int ld_b = kdim;
    std::vector<__nv_fp8_e4m3> h_A, h_B;
    FloatToFp8ColMajorKM(A_row, M, K, h_A, ld_a);
    FloatToFp8ColMajorKN(B_row, K, N, h_B, ld_b);

    __nv_fp8_e4m3* d_A = nullptr;
    __nv_fp8_e4m3* d_B = nullptr;
    float* d_C = nullptr;
    float* d_D = nullptr;
    float* d_scale_a = nullptr;
    float* d_scale_b = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, static_cast<size_t>(ld_a) * M * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&d_B, static_cast<size_t>(ld_b) * N * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&d_C, static_cast<size_t>(M) * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_D, static_cast<size_t>(M) * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scale_a, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scale_b, sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), static_cast<size_t>(ld_a) * M * sizeof(__nv_fp8_e4m3),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), static_cast<size_t>(ld_b) * N * sizeof(__nv_fp8_e4m3),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, static_cast<size_t>(M) * N * sizeof(float)));
    float one_h = 1.0f;
    CHECK_CUDA(cudaMemcpy(d_scale_a, &one_h, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scale_b, &one_h, sizeof(float), cudaMemcpyHostToDevice));

    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t Ad = nullptr, Bd = nullptr, Cd = nullptr, Dd = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &kTransA, sizeof(kTransA)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &kTransB, sizeof(kTransB)));

    float* scale_a_ptr = d_scale_a;
    float* scale_b_ptr = d_scale_b;
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a_ptr,
                                                  sizeof(scale_a_ptr)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b_ptr,
                                                  sizeof(scale_b_ptr)));

    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8F_E4M3,
        static_cast<uint64_t>(kTransA == CUBLAS_OP_N ? m : kdim),
        static_cast<uint64_t>(kTransA == CUBLAS_OP_N ? kdim : m), ld_a));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8F_E4M3,
        static_cast<uint64_t>(kTransB == CUBLAS_OP_N ? kdim : n),
        static_cast<uint64_t>(kTransB == CUBLAS_OP_N ? n : kdim), ld_b));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&Cd, CUDA_R_32F, static_cast<uint64_t>(m),
                                              static_cast<uint64_t>(n), m));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&Dd, CUDA_R_32F, static_cast<uint64_t>(m),
                                              static_cast<uint64_t>(n), m));

    CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&pref));
    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                        &kWorkspaceBytes, sizeof(kWorkspaceBytes)));

    constexpr int kMaxAlgos = 32;
    int returned = 0;
    std::vector<cublasLtMatmulHeuristicResult_t> heur(kMaxAlgos);
    CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(lt, opDesc, Ad, Bd, Cd, Dd, pref, kMaxAlgos, heur.data(), &returned));

    float alpha = 1.0f;
    float beta = 0.0f;
    cudaStream_t stream = nullptr;

    auto try_matmul = [&](const cublasLtMatmulAlgo_t& algo) -> cublasStatus_t {
      return cublasLtMatmul(lt, opDesc, &alpha, d_A, Ad, d_B, Bd, &beta, d_C, Cd, d_D, Dd, &algo, workspace,
                            kWorkspaceBytes, stream);
    };

    int algo_idx = -1;
    for (int a = 0; a < returned; ++a) {
      if (try_matmul(heur[static_cast<size_t>(a)].algo) == CUBLAS_STATUS_SUCCESS) {
        algo_idx = a;
        break;
      }
    }

    std::string note = (algo_idx >= 0) ? ("algo_idx=" + std::to_string(algo_idx)) : std::string("NO_ALGO");

    float ms = 0.0f;
    if (algo_idx >= 0) {
      const cublasLtMatmulAlgo_t& algo = heur[static_cast<size_t>(algo_idx)].algo;
      for (int w = 0; w < kWarmup; ++w) {
        CHECK_CUBLASLT(try_matmul(algo));
      }
      CHECK_CUDA(cudaDeviceSynchronize());

      cudaEvent_t ev0, ev1;
      CHECK_CUDA(cudaEventCreate(&ev0));
      CHECK_CUDA(cudaEventCreate(&ev1));
      CHECK_CUDA(cudaEventRecord(ev0, stream));
      for (int r = 0; r < kRepeat; ++r) {
        CHECK_CUBLASLT(try_matmul(algo));
      }
      CHECK_CUDA(cudaEventRecord(ev1, stream));
      CHECK_CUDA(cudaEventSynchronize(ev1));
      CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));
      ms /= static_cast<float>(kRepeat);
      CHECK_CUDA(cudaEventDestroy(ev0));
      CHECK_CUDA(cudaEventDestroy(ev1));
    }

    std::vector<float> D_row;
    if (algo_idx >= 0) {
      std::vector<float> D_col(static_cast<size_t>(M) * N);
      CHECK_CUDA(cudaMemcpy(D_col.data(), d_D, D_col.size() * sizeof(float), cudaMemcpyDeviceToHost));
      ColMajorMNToRowMajor(D_col.data(), M, N, m, D_row);
    }

    bool ok = true;
    double max_abs_diff = 0.0;
    const char* check = "SKIP";
    if (algo_idx >= 0 && do_cpu_verify) {
      ok = common::CheckEqual(C_cpu, D_row, 0.5f);
      max_abs_diff = common::MaxAbsDiff(C_cpu, D_row);
      check = ok ? "PASS" : "FAIL";
    } else if (algo_idx < 0) {
      check = "NO_ALGO";
      ok = false;
    } else {
      check = "SKIP";
    }

    const double gflops = (ms > 0.0f) ? (2.0 * M * N * K / (ms * 1e6)) : 0.0;

    std::cout << M << "x" << N << "x" << K << " | " << std::fixed << std::setprecision(4) << ms << " ms"
              << " | " << std::setprecision(1) << gflops << " GFLOP/s"
              << " | " << check << " | " << note << "\n";

    ofs << ci << ",cublaslt_fp8_e4m3," << M << "," << N << "," << K << "," << ms << "," << gflops << ","
        << max_abs_diff << "," << check << "," << note << "\n";

    if (pref) CHECK_CUBLASLT(cublasLtMatmulPreferenceDestroy(pref));
    if (Dd) CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(Dd));
    if (Cd) CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(Cd));
    if (Bd) CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(Bd));
    if (Ad) CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(Ad));
    if (opDesc) CHECK_CUBLASLT(cublasLtMatmulDescDestroy(opDesc));

    CHECK_CUDA(cudaFree(d_scale_b));
    CHECK_CUDA(cudaFree(d_scale_a));
    CHECK_CUDA(cudaFree(d_D));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_A));
  }

  CHECK_CUDA(cudaFree(workspace));
  CHECK_CUBLASLT(cublasLtDestroy(lt));
  return 0;
}
