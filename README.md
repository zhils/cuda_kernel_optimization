# CUDA Kernel Optimization

Progressive optimization of core deep learning operators from naive baselines to near-peak-hardware-utilization kernels, benchmarked against cuBLAS/cuDNN on **RTX 5060 Ti (Blackwell, sm_120)**.

## Performance Summary

<!-- TODO: fill in after running benchmarks on your GPU -->

| Operator | Bottleneck | Naive → Best | vs cuBLAS/cuDNN | Peak HW Utilization |
|----------|-----------|-------------|-----------------|---------------------|
| **GEMM** | Compute | _x | _% of cuBLAS | _% compute (GFLOPS) |
| **Softmax** | Memory | _x | _% of cuDNN | _% DRAM bandwidth |
| **RMSNorm** | Memory | _x | _% of cuDNN | _% DRAM bandwidth |

## Optimization Methodology

Each operator follows a systematic optimization pipeline:

```
Theoretical Analysis → Baseline → Profile (Nsight) → Optimize → Validate → Repeat
```

1. **Roofline Analysis** — compute arithmetic intensity, classify as compute-bound or memory-bound, calculate theoretical performance ceiling
2. **Naive Baseline** — correct but unoptimized CUDA kernel
3. **Nsight Compute Profiling** — identify actual bottleneck: warp stalls, memory throughput, occupancy, instruction mix
4. **Progressive Optimization** — 3-5 versions, each targeting a specific bottleneck identified by profiling
5. **Validation** — correctness check against CPU reference, performance comparison against cuBLAS/cuDNN, report achieved % of theoretical peak

## Operators

### GEMM (General Matrix Multiply) — Compute Bound

[Detailed analysis →](gemm/README.md)

| Version | Optimization | Key Technique |
|---------|-------------|---------------|
| V0 | Naive | One thread per output element |
| V1 | Shared memory tiling | 16×16 tile, data reuse in SMEM |
| V2 | Bank conflict elimination | +1 padding, `__ldg`, `#pragma unroll` |
| V3 | Thread-level tiling | 4×4 output per thread, register reuse |
| V4 | Tensor Core | WMMA API (fp32 accumulate) |
| Ref | cuBLAS | `cublasSgemm` baseline |

### Softmax — Memory Bound

[Detailed analysis →](softmax/README.md)

| Version | Optimization | Key Technique |
|---------|-------------|---------------|
| V0 | Naive | Single thread per row, 3-pass |
| V1 | Shared memory reduction | Block-level parallel max/sum |
| V2 | Online softmax | Single-pass algorithm (Milakov 2018) |
| V3 | Warp shuffle + vectorized | `__shfl_sync` reduction, `float4` loads |
| Ref | cuDNN | `cudnnSoftmaxForward` baseline |

### RMSNorm — Memory Bound

[Detailed analysis →](rmsnorm/README.md)

| Version | Optimization | Key Technique |
|---------|-------------|---------------|
| V0 | Naive | Single thread per row |
| V1 | Warp reduction | `__shfl_sync` for sq_sum |
| V2 | Vectorized | `float4` loads + cross-warp reduction |
| V3 | Fused | All threads write output, `__fmul_rn` |
| Ref | cuDNN | `cudnnNormalizationForward` baseline |

## Build & Run

```bash
# Build all targets
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=120
make -j$(nproc)

# Run individual kernel (for Nsight profiling)
cd ..
./build/bin/gemm_v0_naive
./build/bin/gemm_v2_bank_conflict_free

# Run full comparison benchmark
./build/bin/gemm_benchmark_all
./build/bin/softmax_benchmark_all
./build/bin/rmsnorm_benchmark_all

# Profile with Nsight Compute
ncu --set full ./build/bin/gemm_v0_naive
ncu --set full ./build/bin/gemm_v2_bank_conflict_free
```

## Environment

| Item | Specification |
|------|---------------|
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB |
| Architecture | Blackwell (sm_120) |
| CUDA Toolkit | 12.9 |
| cuBLAS / cuDNN | Latest (bundled with toolkit) |
| Timing | `cudaEvent`, trimmed mean of 10 iterations, 3 warmup runs |

## Project Structure

```
├── common/              Shared benchmark utilities (test cases, timing, verification)
├── gemm/                GEMM: V0–V4 + cuBLAS ref + benchmark
├── softmax/             Softmax: V0–V3 + cuDNN ref + benchmark
├── rmsnorm/             RMSNorm: V0–V3 + cuDNN ref + benchmark
├── docs/                Environment & methodology documentation
└── CMakeLists.txt       Top-level build
```
