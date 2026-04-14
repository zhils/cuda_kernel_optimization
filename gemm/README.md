# GEMM — General Matrix Multiply

## Mathematical Definition

```
C[m, n] = Σ_{k=0}^{K-1} A[m, k] × B[k, n]

A ∈ R^{M×K},  B ∈ R^{K×N},  C ∈ R^{M×N}
```

## Theoretical Performance Analysis

### Arithmetic Intensity

| Metric | Formula | Value (square N×N) |
|--------|---------|-------------------|
| Data movement | (M×K + K×N + M×N) × 4 bytes | 12N² bytes |
| Computation | 2 × M × N × K FLOPs | 2N³ FLOPs |
| Arithmetic intensity | 2N³ / 12N² | **N/6 FLOP/Byte** |

### Roofline Classification

RTX 5060 Ti specifications:
- Peak FP32 compute: ~XX TFLOPS
- DRAM bandwidth: ~XX GB/s
- Balance point (ridge point): ~XX FLOP/Byte

| Matrix Size (N×N) | Arith. Intensity | Classification |
|--------------------|-----------------|----------------|
| 128 | ~21 | Near ridge point |
| 512 | ~85 | **Compute bound** |
| 1024 | ~170 | **Compute bound** |
| 4096 | ~683 | **Compute bound** |

For typical deep learning shapes (N > 256), GEMM is **compute bound**.
Optimization should focus on maximizing FMA utilization and minimizing warp stalls.

## Optimization Versions

### V0: Naive Baseline

**File:** `gemm_v0_naive.cu`

Each thread computes one element of C, reading an entire row of A and column of B from global memory.

- **Problem:** Massive redundant global memory reads — each element of A/B is loaded M or N times
- **Nsight diagnosis:** Warp stalls dominated by `Stall Long Scoreboard` (memory latency) and `Stall LG Throttle` (LSU saturation)

<!-- TODO: paste Nsight screenshot and actual stall percentages -->

### V1: Shared Memory Tiling (16×16)

**File:** `gemm_v1_tiled_smem.cu`

Cooperative loading of 16×16 tiles of A and B into shared memory, then computing partial sums from fast SMEM.

- **Key insight:** Each element of A/B is loaded from global memory only once per tile, reused 16 times from SMEM
- **Data reuse ratio:** 16× compared to naive
- **Expected improvement:** Significant reduction in global memory traffic

<!-- TODO: Nsight before/after showing memory throughput change -->

### V2: Bank Conflict Elimination + Read-Only Cache

**File:** `gemm_v2_bank_conflict_free.cu`

- `[TILE][TILE+1]` padding eliminates shared memory bank conflicts
- `__ldg()` routes global loads through read-only texture cache
- `#pragma unroll` on inner loop enables compiler to pipeline FMAs
- `__restrict__` enables more aggressive compiler optimization

<!-- TODO: Nsight showing bank conflict reduction -->

### V3: Thread-Level Tiling (4×4 per thread)

**File:** `gemm_v3_thread_tiling.cu`

Each thread computes a 4×4 sub-block of C, storing partial sums in registers.

- **Key insight:** 16 FMA operations per shared memory load → much higher arithmetic intensity at the thread level
- **Register pressure:** 16 accumulators per thread in registers (fast, no memory access)
- **Thread count:** 8×8 = 64 threads per block (vs 16×16 = 256), but each thread does 16× more work

<!-- TODO: Nsight showing improved compute utilization -->

### V4: Tensor Core (WMMA)

**File:** `benchmark_all.cu` (GemmTensorCoreKernel)

Uses NVIDIA WMMA (Warp Matrix Multiply-Accumulate) API for hardware-accelerated matrix multiply.

- `nvcuda::wmma::mma_sync` issues a single instruction that computes a 16×16×16 matrix multiply
- FP32 accumulation with potential FP16 inputs for even higher throughput

<!-- TODO: WMMA performance numbers -->

### cuBLAS Reference

**File:** `gemm_cublas_ref.cu`

`cublasSgemm` — NVIDIA's production-grade GEMM implementation. This is the performance target.

## Performance Results

<!-- TODO: fill in after running gemm_benchmark_all -->

| M | N | K | Naive (ms) | V1 (ms) | V2 (ms) | V3 (ms) | cuBLAS (ms) | V3/cuBLAS |
|---|---|---|-----------|---------|---------|---------|-------------|-----------|
| 128 | 128 | 128 | | | | | | |
| 512 | 512 | 512 | | | | | | |
| 1024 | 1024 | 1024 | | | | | | |
| 2048 | 2048 | 2048 | | | | | | |
| 4096 | 4096 | 4096 | | | | | | |

## NVIDIA Reference API

- **cuBLAS:** `cublasSgemm` / `cublasGemmEx`
- **cuBLASLt:** `cublasLtMatmul` (more flexible, auto-tuning)
- **CUTLASS:** Template-based high-performance GEMM library
