#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include "common/benchmark.h"
#include "common/cuda_utils.h"

namespace gemm_v5 {
namespace wmma = nvcuda::wmma;
constexpr int kWM = 16, kWN = 16, kWK = 8, kTK = 16;

// --- Kernel S: 32x32 CTA, 32 threads, 1 warp, 2x2 WMMA ---
__global__ __launch_bounds__(32, 16) void GemmS(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {
  extern __shared__ float sm[];
  float *a0=sm,*a1=sm+32*16,*b0=sm+2*32*16,*b1=sm+2*32*16+16*32;
  int tid=threadIdx.x;
  wmma::fragment<wmma::accumulator,16,16,8,float> cf[2][2];
  #pragma unroll
  for(int i=0;i<2;++i)
    #pragma unroll
    for(int j=0;j<2;++j) wmma::fill_fragment(cf[i][j],0.0f);
  int nkt=(K+15)/16;
  auto ld=[&](int ks,float*as,float*bs){
    int tf=(32*16)/4,pp=(tf+31)/32;
    for(int l=0;l<pp;++l){int idx=tid*pp+l;if(idx>=tf)continue;int r=idx/4,ko=(idx%4)*4,gr=blockIdx.y*32+r,gk=ks+ko;float*d=as+r*16+ko;
      if(gr<M&&gk+3<K){__pipeline_memcpy_async(d,&A[size_t(gr)*K+gk],16);}else{d[0]=(gr<M&&gk<K)?A[size_t(gr)*K+gk]:0;d[1]=(gr<M&&gk+1<K)?A[size_t(gr)*K+gk+1]:0;d[2]=(gr<M&&gk+2<K)?A[size_t(gr)*K+gk+2]:0;d[3]=(gr<M&&gk+3<K)?A[size_t(gr)*K+gk+3]:0;}}
    tf=(16*32)/4,pp=(tf+31)/32;
    for(int l=0;l<pp;++l){int idx=tid*pp+l;if(idx>=tf)continue;int ki=idx/8,co=(idx%8)*4,gk=ks+ki,gc=blockIdx.x*32+co;float*d=bs+ki*32+co;
      if(gk<K&&gc+3<N){__pipeline_memcpy_async(d,&B[size_t(gk)*N+gc],16);}else{d[0]=(gk<K&&gc<N)?B[size_t(gk)*N+gc]:0;d[1]=(gk<K&&gc+1<N)?B[size_t(gk)*N+gc+1]:0;d[2]=(gk<K&&gc+2<N)?B[size_t(gk)*N+gc+2]:0;d[3]=(gk<K&&gc+3<N)?B[size_t(gk)*N+gc+3]:0;}}};
  ld(0,a0,b0);__pipeline_commit();__pipeline_wait_prior(0);__syncthreads();
  for(int t=0;t<nkt;++t){
    float*ar=(t&1)?a1:a0,*br=(t&1)?b1:b0;
    if(t+1<nkt){float*aw=(t&1)?a0:a1,*bw=(t&1)?b0:b1;ld((t+1)*16,aw,bw);__pipeline_commit();}
    for(int kk=0;kk<16;kk+=8){
      wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> af[2];
      #pragma unroll
      for(int i=0;i<2;++i)wmma::load_matrix_sync(af[i],ar+(i*16)*16+kk,16);
      for(int j=0;j<2;++j){
        wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bf;
        wmma::load_matrix_sync(bf,br+kk*32+j*16,32);
        #pragma unroll
        for(int i=0;i<2;++i)wmma::mma_sync(cf[i][j],af[i],bf,cf[i][j]);}}
    if(t+1<nkt)__pipeline_wait_prior(0);__syncthreads();}
  #pragma unroll
  for(int i=0;i<2;++i)
    #pragma unroll
    for(int j=0;j<2;++j){int r=blockIdx.y*32+i*16,c=blockIdx.x*32+j*16;
    if(r+16<=M&&c+16<=N)wmma::store_matrix_sync(C+size_t(r)*N+c,cf[i][j],N,wmma::mem_row_major);}
}

// --- Kernel M: 64x64 CTA, 128 threads, 4 warps, 2x2 WMMA/warp ---
__global__ __launch_bounds__(128, 4) void GemmM(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {
  extern __shared__ float sm[];
  float *a0=sm,*a1=sm+64*16,*b0=sm+2*64*16,*b1=sm+2*64*16+16*64;
  int tid=threadIdx.y*32+threadIdx.x,wid=tid/32,wm=wid/2,wn=wid%2;
  wmma::fragment<wmma::accumulator,16,16,8,float> cf[2][2];
  #pragma unroll
  for(int i=0;i<2;++i)
    #pragma unroll
  for(int j=0;j<2;++j) wmma::fill_fragment(cf[i][j],0.0f);
  int nkt=(K+15)/16;
  auto ld=[&](int ks,float*as,float*bs){
    int tf=(64*16)/4,pp=(tf+127)/128;
    for(int l=0;l<pp;++l){int idx=tid*pp+l;if(idx>=tf)continue;int r=idx/4,ko=(idx%4)*4,gr=blockIdx.y*64+r,gk=ks+ko;float*d=as+r*16+ko;
      if(gr<M&&gk+3<K){__pipeline_memcpy_async(d,&A[size_t(gr)*K+gk],16);}else{d[0]=(gr<M&&gk<K)?A[size_t(gr)*K+gk]:0;d[1]=(gr<M&&gk+1<K)?A[size_t(gr)*K+gk+1]:0;d[2]=(gr<M&&gk+2<K)?A[size_t(gr)*K+gk+2]:0;d[3]=(gr<M&&gk+3<K)?A[size_t(gr)*K+gk+3]:0;}}
    tf=(16*64)/4,pp=(tf+127)/128;
    for(int l=0;l<pp;++l){int idx=tid*pp+l;if(idx>=tf)continue;int ki=idx/16,co=(idx%16)*4,gk=ks+ki,gc=blockIdx.x*64+co;float*d=bs+ki*64+co;
      if(gk<K&&gc+3<N){__pipeline_memcpy_async(d,&B[size_t(gk)*N+gc],16);}else{d[0]=(gk<K&&gc<N)?B[size_t(gk)*N+gc]:0;d[1]=(gk<K&&gc+1<N)?B[size_t(gk)*N+gc+1]:0;d[2]=(gk<K&&gc+2<N)?B[size_t(gk)*N+gc+2]:0;d[3]=(gk<K&&gc+3<N)?B[size_t(gk)*N+gc+3]:0;}}};
  ld(0,a0,b0);__pipeline_commit();__pipeline_wait_prior(0);__syncthreads();
  for(int t=0;t<nkt;++t){
    float*ar=(t&1)?a1:a0,*br=(t&1)?b1:b0;
    if(t+1<nkt){float*aw=(t&1)?a0:a1,*bw=(t&1)?b0:b1;ld((t+1)*16,aw,bw);__pipeline_commit();}
    for(int kk=0;kk<16;kk+=8){
      wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> af[2];
      #pragma unroll
      for(int i=0;i<2;++i)wmma::load_matrix_sync(af[i],ar+(wm*32+i*16)*16+kk,16);
      for(int j=0;j<2;++j){
        wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bf;
        wmma::load_matrix_sync(bf,br+kk*64+wn*32+j*16,64);
        #pragma unroll
        for(int i=0;i<2;++i)wmma::mma_sync(cf[i][j],af[i],bf,cf[i][j]);}}
    if(t+1<nkt)__pipeline_wait_prior(0);__syncthreads();}
  #pragma unroll
  for(int i=0;i<2;++i)
    #pragma unroll
  for(int j=0;j<2;++j){int r=blockIdx.y*64+wm*32+i*16,c=blockIdx.x*64+wn*32+j*16;
    if(r+16<=M&&c+16<=N)wmma::store_matrix_sync(C+size_t(r)*N+c,cf[i][j],N,wmma::mem_row_major);}
}

// --- Kernel L: 128x128 CTA, 256 threads, 8 warps, 4x2 WMMA/warp ---
__global__ __launch_bounds__(256, 2) void GemmL(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {
  extern __shared__ float sm[];
  float *a0=sm,*a1=sm+128*16,*b0=sm+2*128*16,*b1=sm+2*128*16+16*128;
  int tid=threadIdx.y*32+threadIdx.x,wid=tid/32,wm=wid/4,wn=wid%4;
  wmma::fragment<wmma::accumulator,16,16,8,float> cf[4][2];
  #pragma unroll
  for(int i=0;i<4;++i)
    #pragma unroll
  for(int j=0;j<2;++j) wmma::fill_fragment(cf[i][j],0.0f);
  int nkt=(K+15)/16;
  auto ld=[&](int ks,float*as,float*bs){
    int tf=(128*16)/4,pp=(tf+255)/256;
    for(int l=0;l<pp;++l){int idx=tid*pp+l;if(idx>=tf)continue;int r=idx/4,ko=(idx%4)*4,gr=blockIdx.y*128+r,gk=ks+ko;float*d=as+r*16+ko;
      if(gr<M&&gk+3<K){__pipeline_memcpy_async(d,&A[size_t(gr)*K+gk],16);}else{d[0]=(gr<M&&gk<K)?A[size_t(gr)*K+gk]:0;d[1]=(gr<M&&gk+1<K)?A[size_t(gr)*K+gk+1]:0;d[2]=(gr<M&&gk+2<K)?A[size_t(gr)*K+gk+2]:0;d[3]=(gr<M&&gk+3<K)?A[size_t(gr)*K+gk+3]:0;}}
    tf=(16*128)/4,pp=(tf+255)/256;
    for(int l=0;l<pp;++l){int idx=tid*pp+l;if(idx>=tf)continue;int ki=idx/32,co=(idx%32)*4,gk=ks+ki,gc=blockIdx.x*128+co;float*d=bs+ki*128+co;
      if(gk<K&&gc+3<N){__pipeline_memcpy_async(d,&B[size_t(gk)*N+gc],16);}else{d[0]=(gk<K&&gc<N)?B[size_t(gk)*N+gc]:0;d[1]=(gk<K&&gc+1<N)?B[size_t(gk)*N+gc+1]:0;d[2]=(gk<K&&gc+2<N)?B[size_t(gk)*N+gc+2]:0;d[3]=(gk<K&&gc+3<N)?B[size_t(gk)*N+gc+3]:0;}}};
  ld(0,a0,b0);__pipeline_commit();__pipeline_wait_prior(0);__syncthreads();
  for(int t=0;t<nkt;++t){
    float*ar=(t&1)?a1:a0,*br=(t&1)?b1:b0;
    if(t+1<nkt){float*aw=(t&1)?a0:a1,*bw=(t&1)?b0:b1;ld((t+1)*16,aw,bw);__pipeline_commit();}
    for(int kk=0;kk<16;kk+=8){
      wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> af[4];
      #pragma unroll
      for(int i=0;i<4;++i)wmma::load_matrix_sync(af[i],ar+(wm*64+i*16)*16+kk,16);
      for(int j=0;j<2;++j){
        wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bf;
        wmma::load_matrix_sync(bf,br+kk*128+wn*32+j*16,128);
        #pragma unroll
        for(int i=0;i<4;++i)wmma::mma_sync(cf[i][j],af[i],bf,cf[i][j]);}}
    if(t+1<nkt)__pipeline_wait_prior(0);__syncthreads();}
  #pragma unroll
  for(int i=0;i<4;++i)
    #pragma unroll
  for(int j=0;j<2;++j){int r=blockIdx.y*128+wm*64+i*16,c=blockIdx.x*128+wn*32+j*16;
    if(r+16<=M&&c+16<=N)wmma::store_matrix_sync(C+size_t(r)*N+c,cf[i][j],N,wmma::mem_row_major);}
}

// --- Unified dispatcher ---
void Launch(const float* dA, const float* dB, float* dC, int M, int N, int K) {
  int md = std::min({M, N, K});
  if (md <= 128) {
    dim3 g((N+31)/32,(M+31)/32); dim3 b(32,1);
    CHECK_CUDA(cudaFuncSetAttribute(GemmS,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(float)*2*(32*16+16*32)));
    GemmS<<<g,b,sizeof(float)*2*(32*16+16*32)>>>(dA,dB,dC,M,N,K);
  } else if (md <= 512) {
    dim3 g((N+63)/64,(M+63)/64); dim3 b(32,4);
    CHECK_CUDA(cudaFuncSetAttribute(GemmM,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(float)*2*(64*16+16*64)));
    GemmM<<<g,b,sizeof(float)*2*(64*16+16*64)>>>(dA,dB,dC,M,N,K);
  } else {
    dim3 g((N+127)/128,(M+127)/128); dim3 b(32,8);
    CHECK_CUDA(cudaFuncSetAttribute(GemmL,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(float)*2*(128*16+16*128)));
    GemmL<<<g,b,sizeof(float)*2*(128*16+16*128)>>>(dA,dB,dC,M,N,K);
  }
}
}

static void GemmCPU(const float* A, const float* B, float* C, int M, int N, int K) {
  for(int r=0;r<M;++r)for(int c=0;c<N;++c){float s=0;for(int k=0;k<K;++k)s+=A[size_t(r)*K+k]*B[size_t(k)*N+c];C[size_t(r)*N+c]=s;}
}

int main() {
  constexpr int kR=10;
  auto cases=common::LoadOrCreateTestCasesCsv("data/gemm/test_cases.csv");
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/gemm_v5_results.csv");
  ofs<<"id,group,M,N,K,gpu_ms,gflops,max_abs_diff,check\n";
  for(auto& tc:cases){
    int M=tc.rows,N=tc.cols,K=M;bool aligned=(M%16==0)&&(N%16==0)&&(K%16==0);
    std::vector<float> A(size_t(M)*K),B(size_t(K)*N),Cc(size_t(M)*N),Cg(size_t(M)*N);
    common::InitMatrix(A,M,K);common::InitMatrix(B,K,N);
    if(M<=1024&&N<=1024)GemmCPU(A.data(),B.data(),Cc.data(),M,N,K);
    float ms=0;
    if(aligned){
      float*dA,*dB,*dC;
      CHECK_CUDA(cudaMalloc(&dA,A.size()*4));CHECK_CUDA(cudaMalloc(&dB,B.size()*4));CHECK_CUDA(cudaMalloc(&dC,Cg.size()*4));
      CHECK_CUDA(cudaMemcpy(dA,A.data(),A.size()*4,cudaMemcpyHostToDevice));CHECK_CUDA(cudaMemcpy(dB,B.data(),B.size()*4,cudaMemcpyHostToDevice));
      gemm_v5::Launch(dA,dB,dC,M,N,K);CHECK_CUDA(cudaDeviceSynchronize());
      cudaEvent_t s,e;CHECK_CUDA(cudaEventCreate(&s));CHECK_CUDA(cudaEventCreate(&e));
      CHECK_CUDA(cudaEventRecord(s));
      for(int rep=0;rep<kR;++rep)gemm_v5::Launch(dA,dB,dC,M,N,K);
      CHECK_CUDA(cudaEventRecord(e));CHECK_CUDA(cudaEventSynchronize(e));CHECK_CUDA(cudaEventElapsedTime(&ms,s,e));ms/=kR;
      CHECK_CUDA(cudaMemcpy(Cg.data(),dC,Cg.size()*4,cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaEventDestroy(s));CHECK_CUDA(cudaEventDestroy(e));CHECK_CUDA(cudaFree(dA));CHECK_CUDA(cudaFree(dB));CHECK_CUDA(cudaFree(dC));
    }
    bool ok=true;double diff=0;const char*chk="SKIP";
    if(aligned&&M<=1024&&N<=1024){ok=common::CheckEqual(Cc,Cg,2e-1f);diff=common::MaxAbsDiff(Cc,Cg);chk=ok?"PASS":"FAIL";}
    double gflops=(ms>0)?(2.0*M*N*K/(ms*1e6)):0;
    std::cout<<M<<"x"<<N<<"x"<<K<<" | "<<std::fixed<<std::setprecision(4)<<ms<<" ms | "<<std::setprecision(1)<<gflops<<" GFLOP/s | "<<chk<<"\n";
    ofs<<0<<",gemm_v5,"<<M<<","<<N<<","<<K<<","<<ms<<","<<gflops<<","<<diff<<","<<chk<<"\n";
  }
}
