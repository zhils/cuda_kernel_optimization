// GEMM All Variants Comparison
//
// Unified benchmark comparing all GEMM implementations:
//   Hand-written: gemm_v3 (FP32), gemm_fp16 (FP16 WMMA),
//                 gemm_fp16 (FP16 WMMA), gemm_bf16 (BF16 WMMA)
//   cuBLAS:       SGEMM (FP32), HGEMM (FP16), BF16 GEMM
//   CUTLASS:      FP16/BF16 (conditional, needs CUTLASS_ROOT)
//
// Compile:
//   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
//   cmake --build build --target gemm_all_compare
//
// With CUTLASS:
//   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUTLASS_ROOT=~/cutlass
//   cmake --build build --target gemm_all_compare

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "common/benchmark.h"
#include "common/cuda_utils.h"

#ifdef CUTLASS_ROOT
  #include "cutlass/cutlass.h"
  #include "cutlass/gemm/device/gemm.h"
  #define HAS_CUTLASS 1
#else
  #define HAS_CUTLASS 0
#endif

#define CHECK_CUBLAS(call) do { \
  cublasStatus_t s__ = (call); \
  if (s__ != CUBLAS_STATUS_SUCCESS) { \
    std::cerr << "cuBLAS error " << int(s__) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

// ============================================================
// Include hand-written kernel source files (single TU, no device linking needed)
// Rename conflicting static helper functions via preprocessor.
// ============================================================
#define __CUDA_AMPERE_MMA__
#include "gemm_v3.cu"
#define PrecisionMetrics PrecisionMetrics_fp16
#define GemmCPU_FP32 GemmCPU_FP32_fp16
#include "gemm_fp16.cu"
#undef PrecisionMetrics
#undef ComputeMetrics
#undef GemmCPU_FP32
#define PrecisionMetrics PrecisionMetrics_bf16
#define ComputeMetrics ComputeMetrics_bf16
#define GemmCPU_FP32 GemmCPU_FP32_bf16
#include "gemm_bf16.cu"
#undef PrecisionMetrics
#undef ComputeMetrics
#undef GemmCPU_FP32

// ============================================================
// Result struct & CPU reference
// ============================================================
struct Result {
  std::string name; float ms; double gflops; double err; bool pass; bool skip;
};

static void GemmCPU_Ref(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < N; ++c) {
      float s = 0;
      for (int k = 0; k < K; ++k) s += A[size_t(r)*K+k] * B[size_t(k)*N+c];
      C[size_t(r)*N+c] = s;
    }
}

// ============================================================
// Benchmark helpers
// ============================================================

static Result bench_v3(const float* A, const float* B, const float* Cref,
                       int M, int N, int K, int rep, int maxCpu) {
  Result r{"gemm_v3 (FP32 CUDA Core)",0,0,0,false,false};
  if (M%kBlockM || N%kBlockN || K%kTileK) { r.skip=true; return r; }
  size_t szA=size_t(M)*K, szB=size_t(K)*N, szC=size_t(M)*N;
  float *dA, *dB, *dC;
  cudaMalloc(&dA,szA*4); cudaMalloc(&dB,szB*4); cudaMalloc(&dC,szC*4);
  cudaMemcpy(dA,A,szA*4,cudaMemcpyHostToDevice); cudaMemcpy(dB,B,szB*4,cudaMemcpyHostToDevice);
  dim3 b(kBlockThreadsX,kBlockThreadsY), g((N+kBlockN-1)/kBlockN,(M+kBlockM-1)/kBlockM);
  cudaFuncSetAttribute(GemmV3Kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(float)*2*(kSmemABuf+kSmemBBuf));
  GemmV3Kernel<<<g,b,sizeof(float)*2*(kSmemABuf+kSmemBBuf)>>>(dA,dB,dC,M,N,K); cudaDeviceSynchronize();
  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s);
  for (int i=0;i<rep;++i) GemmV3Kernel<<<g,b,sizeof(float)*2*(kSmemABuf+kSmemBBuf)>>>(dA,dB,dC,M,N,K);
  cudaEventRecord(e); cudaEventSynchronize(e); float ms;
  cudaEventElapsedTime(&ms,s,e); r.ms=ms/rep;
  if (M<=maxCpu&&N<=maxCpu) {
    std::vector<float> Cg(szC); cudaMemcpy(Cg.data(),dC,szC*4,cudaMemcpyDeviceToHost);
    r.err=common::MaxAbsDiff(std::vector<float>(Cref,Cref+szC),Cg); r.pass=r.err<1e-2f;
  }
  r.gflops=2.0*M*N*K/(r.ms*1e6);
  cudaEventDestroy(s);cudaEventDestroy(e);cudaFree(dA);cudaFree(dB);cudaFree(dC);
  return r;
}


static Result bench_fp16(const float* A, const float* B, const float* Cref,
                         int M, int N, int K, int rep, int maxCpu) {
  Result r{"gemm_fp16 (FP16 WMMA)",0,0,0,false,false};
  if(M%gemm_fp16::kBlockM||N%gemm_fp16::kBlockN||K%gemm_fp16::kTileK){r.skip=true;return r;}
  size_t szA=size_t(M)*K,szB=size_t(K)*N,szC=size_t(M)*N;
  std::vector<__half>Ah(szA),Bh(szB);
  for(size_t j=0;j<szA;++j)Ah[j]=__float2half(A[j]);
  for(size_t j=0;j<szB;++j)Bh[j]=__float2half(B[j]);
  __half*dA,*dB;float*dC;
  cudaMalloc(&dA,szA*2);cudaMalloc(&dB,szB*2);cudaMalloc(&dC,szC*4);
  cudaMemcpy(dA,Ah.data(),szA*2,cudaMemcpyHostToDevice);
  cudaMemcpy(dB,Bh.data(),szB*2,cudaMemcpyHostToDevice);
  dim3 b(gemm_fp16::kBlockThreadsX,gemm_fp16::kBlockThreadsY),g((N+gemm_fp16::kBlockN-1)/gemm_fp16::kBlockN,(M+gemm_fp16::kBlockM-1)/gemm_fp16::kBlockM);
  cudaFuncSetAttribute(gemm_fp16::GemmFP16Kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,gemm_fp16::kSmemSize);
  gemm_fp16::GemmFP16Kernel<<<g,b,gemm_fp16::kSmemSize>>>(dA,dB,dC,M,N,K);cudaDeviceSynchronize();
  cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);cudaEventRecord(s);
  for(int i=0;i<rep;++i)gemm_fp16::GemmFP16Kernel<<<g,b,gemm_fp16::kSmemSize>>>(dA,dB,dC,M,N,K);
  cudaEventRecord(e);cudaEventSynchronize(e);float ms;
  cudaEventElapsedTime(&ms,s,e);r.ms=ms/rep;
  if(M<=maxCpu&&N<=maxCpu){std::vector<float>Cg(szC);cudaMemcpy(Cg.data(),dC,szC*4,cudaMemcpyDeviceToHost);
    r.err=common::MaxAbsDiff(std::vector<float>(Cref,Cref+szC),Cg);r.pass=r.err<2e-1f;}
  r.gflops=2.0*M*N*K/(r.ms*1e6);
  cudaEventDestroy(s);cudaEventDestroy(e);cudaFree(dA);cudaFree(dB);cudaFree(dC);
  return r;
}

static Result bench_bf16(const float* A, const float* B, const float* Cref,
                         int M, int N, int K, int rep, int maxCpu) {
  Result r{"gemm_bf16 (BF16 WMMA)",0,0,0,false,false};
  if(M%gemm_bf16::kBlockM||N%gemm_bf16::kBlockN||K%gemm_bf16::kTileK){r.skip=true;return r;}
  size_t szA=size_t(M)*K,szB=size_t(K)*N,szC=size_t(M)*N;
  std::vector<__nv_bfloat16>Ab(szA),Bb(szB);
  for(size_t j=0;j<szA;++j)Ab[j]=__float2bfloat16(A[j]);
  for(size_t j=0;j<szB;++j)Bb[j]=__float2bfloat16(B[j]);
  __nv_bfloat16*dA,*dB;float*dC;
  cudaMalloc(&dA,szA*2);cudaMalloc(&dB,szB*2);cudaMalloc(&dC,szC*4);
  cudaMemcpy(dA,Ab.data(),szA*2,cudaMemcpyHostToDevice);
  cudaMemcpy(dB,Bb.data(),szB*2,cudaMemcpyHostToDevice);
  dim3 b(gemm_bf16::kBlockThreadsX,gemm_bf16::kBlockThreadsY),g((N+gemm_bf16::kBlockN-1)/gemm_bf16::kBlockN,(M+gemm_bf16::kBlockM-1)/gemm_bf16::kBlockM);
  cudaFuncSetAttribute(gemm_bf16::GemmBF16Kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,gemm_bf16::kSmemSize);
  gemm_bf16::GemmBF16Kernel<<<g,b,gemm_bf16::kSmemSize>>>(dA,dB,dC,M,N,K);cudaDeviceSynchronize();
  cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);cudaEventRecord(s);
  for(int i=0;i<rep;++i)gemm_bf16::GemmBF16Kernel<<<g,b,gemm_bf16::kSmemSize>>>(dA,dB,dC,M,N,K);
  cudaEventRecord(e);cudaEventSynchronize(e);float ms;
  cudaEventElapsedTime(&ms,s,e);r.ms=ms/rep;
  if(M<=maxCpu&&N<=maxCpu){std::vector<float>Cg(szC);cudaMemcpy(Cg.data(),dC,szC*4,cudaMemcpyDeviceToHost);
    r.err=common::MaxAbsDiff(std::vector<float>(Cref,Cref+szC),Cg);r.pass=r.err<2e-1f;}
  r.gflops=2.0*M*N*K/(r.ms*1e6);
  cudaEventDestroy(s);cudaEventDestroy(e);cudaFree(dA);cudaFree(dB);cudaFree(dC);
  return r;
}

static Result bench_cublas_fp32(cublasHandle_t h, const float* A, const float* B, const float* Cref,
                                int M, int N, int K, int rep, int warm, int maxCpu) {
  Result r{"cuBLAS FP32",0,0,0,false,false};
  size_t szA=size_t(M)*K,szB=size_t(K)*N,szC=size_t(M)*N;
  float *dA,*dB,*dC;cudaMalloc(&dA,szA*4);cudaMalloc(&dB,szB*4);cudaMalloc(&dC,szC*4);
  cudaMemcpy(dA,A,szA*4,cudaMemcpyHostToDevice);cudaMemcpy(dB,B,szB*4,cudaMemcpyHostToDevice);
  float a=1,b=0;
  for(int w=0;w<warm;++w)cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,dB,N,dA,K,&b,dC,N);
  cudaDeviceSynchronize();
  cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);cudaEventRecord(s);
  for(int i=0;i<rep;++i)cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,dB,N,dA,K,&b,dC,N);
  cudaEventRecord(e);cudaEventSynchronize(e);float ms;
  cudaEventElapsedTime(&ms,s,e);r.ms=ms/rep;
  if(M<=maxCpu&&N<=maxCpu){std::vector<float>Cg(szC);cudaMemcpy(Cg.data(),dC,szC*4,cudaMemcpyDeviceToHost);
    r.err=common::MaxAbsDiff(std::vector<float>(Cref,Cref+szC),Cg);r.pass=r.err<1e-2f;}
  r.gflops=2.0*M*N*K/(r.ms*1e6);
  cudaEventDestroy(s);cudaEventDestroy(e);cudaFree(dA);cudaFree(dB);cudaFree(dC);
  return r;
}

static Result bench_cublas_fp16(cublasHandle_t h, const float* A, const float* B, const float* Cref,
                                int M, int N, int K, int rep, int warm, int maxCpu) {
  Result r{"cuBLAS FP16",0,0,0,false,false};
  size_t szA=size_t(M)*K,szB=size_t(K)*N,szC=size_t(M)*N;
  std::vector<__half>Ah(szA),Bh(szB);
  for(size_t j=0;j<szA;++j)Ah[j]=__float2half(A[j]);for(size_t j=0;j<szB;++j)Bh[j]=__float2half(B[j]);
  __half*dA,*dB;float*dC;cudaMalloc(&dA,szA*2);cudaMalloc(&dB,szB*2);cudaMalloc(&dC,szC*4);
  cudaMemcpy(dA,Ah.data(),szA*2,cudaMemcpyHostToDevice);cudaMemcpy(dB,Bh.data(),szB*2,cudaMemcpyHostToDevice);
  float a=1,b=0;
  for(int w=0;w<warm;++w)cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,dB,CUDA_R_16F,N,dA,CUDA_R_16F,K,&b,dC,CUDA_R_32F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaDeviceSynchronize();
  cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);cudaEventRecord(s);
  for(int i=0;i<rep;++i)cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,dB,CUDA_R_16F,N,dA,CUDA_R_16F,K,&b,dC,CUDA_R_32F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaEventRecord(e);cudaEventSynchronize(e);float ms;
  cudaEventElapsedTime(&ms,s,e);r.ms=ms/rep;
  if(M<=maxCpu&&N<=maxCpu){std::vector<float>Cg(szC);cudaMemcpy(Cg.data(),dC,szC*4,cudaMemcpyDeviceToHost);
    r.err=common::MaxAbsDiff(std::vector<float>(Cref,Cref+szC),Cg);r.pass=r.err<2e-1f;}
  r.gflops=2.0*M*N*K/(r.ms*1e6);
  cudaEventDestroy(s);cudaEventDestroy(e);cudaFree(dA);cudaFree(dB);cudaFree(dC);
  return r;
}

static Result bench_cublas_bf16(cublasHandle_t h, const float* A, const float* B, const float* Cref,
                                int M, int N, int K, int rep, int warm, int maxCpu) {
  Result r{"cuBLAS BF16",0,0,0,false,false};
  size_t szA=size_t(M)*K,szB=size_t(K)*N,szC=size_t(M)*N;
  std::vector<__nv_bfloat16>Ab(szA),Bb(szB);
  for(size_t j=0;j<szA;++j)Ab[j]=__float2bfloat16(A[j]);for(size_t j=0;j<szB;++j)Bb[j]=__float2bfloat16(B[j]);
  __nv_bfloat16*dA,*dB;float*dC;cudaMalloc(&dA,szA*2);cudaMalloc(&dB,szB*2);cudaMalloc(&dC,szC*4);
  cudaMemcpy(dA,Ab.data(),szA*2,cudaMemcpyHostToDevice);cudaMemcpy(dB,Bb.data(),szB*2,cudaMemcpyHostToDevice);
  float a=1,b=0;
  for(int w=0;w<warm;++w)cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,dB,CUDA_R_16BF,N,dA,CUDA_R_16BF,K,&b,dC,CUDA_R_32F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaDeviceSynchronize();
  cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);cudaEventRecord(s);
  for(int i=0;i<rep;++i)cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,dB,CUDA_R_16BF,N,dA,CUDA_R_16BF,K,&b,dC,CUDA_R_32F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaEventRecord(e);cudaEventSynchronize(e);float ms;
  cudaEventElapsedTime(&ms,s,e);r.ms=ms/rep;
  if(M<=maxCpu&&N<=maxCpu){std::vector<float>Cg(szC);cudaMemcpy(Cg.data(),dC,szC*4,cudaMemcpyDeviceToHost);
    r.err=common::MaxAbsDiff(std::vector<float>(Cref,Cref+szC),Cg);r.pass=r.err<2e-1f;}
  r.gflops=2.0*M*N*K/(r.ms*1e6);
  cudaEventDestroy(s);cudaEventDestroy(e);cudaFree(dA);cudaFree(dB);cudaFree(dC);
  return r;
}

#if HAS_CUTLASS
template<typename EA, typename ET, typename Arch>
static Result bench_cutlass(const char* label, const float* A, const float* B, const float* Cref,
                            int M, int N, int K, int rep, int warm, int maxCpu) {
  Result r{label,0,0,0,false,false};
  size_t szA=size_t(M)*K,szB=size_t(K)*N,szC=size_t(M)*N;
  std::vector<EA> Ac(szA), Bc(szB);
  for(size_t j=0;j<szA;++j)Ac[j]=EA(A[j]);for(size_t j=0;j<szB;++j)Bc[j]=EA(B[j]);
  EA *dA,*dB;float*dC;
  cudaMalloc(&dA,szA*sizeof(EA));cudaMalloc(&dB,szB*sizeof(EA));cudaMalloc(&dC,szC*4);
  cudaMemcpy(dA,Ac.data(),szA*sizeof(EA),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,Bc.data(),szB*sizeof(EA),cudaMemcpyHostToDevice);
  using Gemm = cutlass::gemm::device::Gemm<EA,cutlass::layout::RowMajor,EA,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,ET,cutlass::arch::OpClassTensorOp,Arch>;
  Gemm gemm;
  typename Gemm::Arguments args{{M,N,K},{dA,typename Gemm::GemmKernel::StrideA(K)},{dB,typename Gemm::GemmKernel::StrideB(N)},{dC,typename Gemm::GemmKernel::StrideC(N)},{dC,typename Gemm::GemmKernel::StrideC(N)},{1.0f,0.0f}};
  auto st=gemm(args);
  if(st!=cutlass::Status::kSuccess){r.skip=true;cudaFree(dA);cudaFree(dB);cudaFree(dC);return r;}
  cudaDeviceSynchronize();
  cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);cudaEventRecord(s);
  for(int i=0;i<rep;++i)gemm(args);
  cudaEventRecord(e);cudaEventSynchronize(e);float ms;
  cudaEventElapsedTime(&ms,s,e);r.ms=ms/rep;
  if(M<=maxCpu&&N<=maxCpu){std::vector<float>Cg(szC);cudaMemcpy(Cg.data(),dC,szC*4,cudaMemcpyDeviceToHost);
    r.err=common::MaxAbsDiff(std::vector<float>(Cref,Cref+szC),Cg);r.pass=r.err<2e-1f;}
  r.gflops=2.0*M*N*K/(r.ms*1e6);
  cudaEventDestroy(s);cudaEventDestroy(e);cudaFree(dA);cudaFree(dB);cudaFree(dC);
  return r;
}
#endif

int main(int argc, char** argv) {
  constexpr int kRep=10, kWarm=3, kMaxCpu=1024;
  std::vector<std::tuple<int,int,int>> shapes = {
    {128,128,128},{256,256,256},{512,512,512},{1024,1024,1024},{4096,4096,4096}
  };
  std::string csv = "data/results/gemm_all_compare_results.csv";
  std::filesystem::create_directories("data/results");

  cudaDeviceProp prop; cudaGetDeviceProperties(&prop,0);
  cublasHandle_t h; cublasCreate(&h);

  std::ofstream ofs(csv);
  ofs << "M,N,K,impl,gpu_ms,gflops,max_abs_err,check\n";

  for (auto [M,N,K] : shapes) {
    size_t szA=size_t(M)*K,szB=size_t(K)*N,szC=size_t(M)*N;
    std::vector<float> A(szA),B(szB),Cref(szC);
    common::InitMatrix(A,M,K); common::InitMatrix(B,K,N);
    if(M<=kMaxCpu&&N<=kMaxCpu) GemmCPU_Ref(A.data(),B.data(),Cref.data(),M,N,K);

    std::vector<Result> results;
    results.push_back(bench_v3(A.data(),B.data(),Cref.data(),M,N,K,kRep,kMaxCpu));
    results.push_back(bench_fp16(A.data(),B.data(),Cref.data(),M,N,K,kRep,kMaxCpu));
    results.push_back(bench_bf16(A.data(),B.data(),Cref.data(),M,N,K,kRep,kMaxCpu));
    results.push_back(bench_cublas_fp32(h,A.data(),B.data(),Cref.data(),M,N,K,kRep,kWarm,kMaxCpu));
    results.push_back(bench_cublas_fp16(h,A.data(),B.data(),Cref.data(),M,N,K,kRep,kWarm,kMaxCpu));
    results.push_back(bench_cublas_bf16(h,A.data(),B.data(),Cref.data(),M,N,K,kRep,kWarm,kMaxCpu));
#if HAS_CUTLASS
    results.push_back(bench_cutlass<cutlass::half_t,float,cutlass::arch::Sm100>("CUTLASS FP16",A.data(),B.data(),Cref.data(),M,N,K,kRep,kWarm,kMaxCpu));
    results.push_back(bench_cutlass<cutlass::bfloat16_t,float,cutlass::arch::Sm100>("CUTLASS BF16",A.data(),B.data(),Cref.data(),M,N,K,kRep,kWarm,kMaxCpu));
#endif

    // find fastest
    float best = 1e30f;
    for(auto& r:results) if(!r.skip&&r.ms>0) best=std::min(best,r.ms);

    std::cout << "\n--- " << M << "x" << N << "x" << K << " ---\n";
    std::cout << std::left << std::setw(28) << "Implementation"
              << std::right << std::setw(12) << "GPU ms" << std::setw(14) << "GFLOPS"
              << std::setw(10) << "Check" << std::setw(10) << "vs best" << "\n";
    std::cout << std::string(74,'-') << "\n";

    for(auto& res:results){
      if(res.skip){std::cout<<std::left<<std::setw(28)<<res.name<<"  (skipped)\n";continue;}
      const char*ck=(M<=kMaxCpu&&N<=kMaxCpu)?(res.pass?"PASS":"FAIL"):"SKIP";
      std::cout<<std::left<<std::setw(28)<<res.name
               <<std::right<<std::fixed<<std::setprecision(4)<<std::setw(12)<<res.ms
               <<std::setprecision(1)<<std::setw(14)<<res.gflops
               <<std::setw(10)<<ck
               <<std::setw(10)<<std::setprecision(3)<<(res.ms/best)<<"x\n";
      ofs<<M<<","<<N<<","<<K<<","<<res.name<<","<<res.ms<<","<<res.gflops<<","<<res.err<<","<<ck<<"\n";
    }
  }

  cublasDestroy(h);
  std::cout << "\nResults saved to " << csv << "\n";
#if !HAS_CUTLASS
  std::cout << "Note: CUTLASS not compiled. Enable via -DCUTLASS_ROOT=~/cutlass\n";
#endif
  return 0;
}
