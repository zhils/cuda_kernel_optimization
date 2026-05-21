#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "common/benchmark.h"
#include "common/cuda_utils.h"
#include "common/kernel_config.h"
// inline CpuGemm defined below

static void GemmCPU(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double sum = 0;
      for (int k = 0; k < K; ++k) sum += static_cast<double>(A[i * K + k]) * B[k * N + j];
      C[i * N + j] = static_cast<float>(sum);
    }
}

constexpr int TM = 16;
constexpr int TN = 16;
constexpr int TK = 16;

__global__ void GemmV1Kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__  C,
    int M,
    int N,
    int K
){
    // 线程索引与边界
    const int row = blockIdx.y * TM;
    const int col = blockIdx.x * TN;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    bool valid = (row + ty < M && col + tx < N);

    // 共享内存
    __shared__ float As[TM][TK];
    __shared__ float Bs[TK][TN];

    int num_k = (K + TK- 1) / TK;
    float sum = 0.0f;
    
    // 主循环：加载 → 同步 → 计算 → 同步
    for(int t = 0; t < num_k; ++t){
        // 加载A到共享内存
        if(t * TK + tx < K){
            As[ty][tx] = A[(row + ty) * K + t * TK + tx];        
        } else {
            As[ty][tx] = 0.0f;
        }

        // 加载B到共享内存
        if(TK * t + ty < K){
            Bs[ty][tx] = B[(ty + t * TK) * N + col + tx];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();

        // 计算乘加
        for(int k = 0; k < TK; ++k){
           sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    // 写回结果
    if(valid){
        C[(row + ty) * N + col + tx] = sum;
    }

}

int main() {
    // 参数与输出文件准备
    constexpr int kRepeat = 10;
    constexpr int kMaxCpuVerifyDim = 512;
    auto cases = common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
    std::filesystem::create_directories("data/results");
    std::ofstream ofs("data/results/gemm_v1_results.csv");
    ofs << "id,group,M,N,K,gpu_ms,gflops,max_abs_diff,check\n";

    for (size_t i = 0; i < cases.size(); ++i) {
        int M = cases[i].rows, N = cases[i].cols, K = M;

        // 生成测试数据
        std::vector<float> A(static_cast<size_t>(M) * K),
                           B(static_cast<size_t>(K) * N),
                           C_cpu(static_cast<size_t>(M) * N),
                           C_gpu(static_cast<size_t>(M) * N);
        common::InitMatrix(A, M, K);
        common::InitMatrix(B, K, N);

        // CPU参考计算
        if (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim) {
            GemmCPU(A.data(), B.data(), C_cpu.data(), M, N, K);
        }

        // 分配GPU内存
        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

        // 拷贝数据到设备
        CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

        // 启动配置与预热
        common::GemmLaunchConfig launch_cfg = common::MakeGemmLaunchConfig(
            M, N, TM, TN, TK, TN, TM);
        GemmV1Kernel<<<launch_cfg.grid, launch_cfg.block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        // 计时循环
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        for (int r = 0; r < kRepeat; ++r) {
            GemmV1Kernel<<<launch_cfg.grid, launch_cfg.block>>>(d_A, d_B, d_C, M, N, K);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        ms /= kRepeat;

        // 拷贝结果回主机
        CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        // 校验
        bool did_verify = (M <= kMaxCpuVerifyDim && N <= kMaxCpuVerifyDim);
        bool ok = true;
        if (did_verify) {
            ok = common::CheckEqual(C_cpu, C_gpu, 1e-3f);
        }

        // 计算并输出结果
        double gflops = (2.0 * M * N * K) / (ms * 1e6);
        const char* check = did_verify ? (ok ? "PASS" : "FAIL") : "NOT_RUN";
        const std::string max_abs_diff =
            did_verify ? std::to_string(common::MaxAbsDiff(C_cpu, C_gpu)) : "";
        std::cout << M << "x" << N << "x" << K << " | " << std::fixed << std::setprecision(4) << ms << " ms"
                  << " | " << std::setprecision(1) << gflops << " GFLOP/s"
                  << " | " << check << "\n";
        ofs << i << ",gemm_v1," << M << "," << N << "," << K << ","
            << ms << "," << gflops << ","
            << max_abs_diff << ","
            << check << "\n";

        // 释放资源
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
    }
    return 0;
}
