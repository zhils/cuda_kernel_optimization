# RMSNorm

## Mathematical Definition

```
y = x / sqrt(mean(x²) + ε) × γ

where mean(x²) = (1/C) × Σ_{i=0}^{C-1} x_i²
```

RMSNorm (Root Mean Square Layer Normalization) is widely used in modern LLMs (LLaMA, Qwen, etc.) as a simpler and faster alternative to LayerNorm — it removes the mean-centering step, computing only the RMS for normalization.

## Theoretical Performance Analysis

### Arithmetic Intensity

| Metric | Formula | Value |
|--------|---------|-------|
| Data movement | Read x + read γ + write y = (2R×C + C) × 4 bytes | ~8RC bytes |
| Computation | R × (C mul + C add + 1 rsqrt + C mul + C mul) | ~4RC FLOPs |
| Arithmetic intensity | 4RC / 8RC | **~0.5 FLOP/Byte** |

### Roofline Classification

Like Softmax, RMSNorm is deeply **memory bound**. The optimization target is maximizing DRAM bandwidth utilization.

### Theoretical Peak

- RTX 5060 Ti DRAM bandwidth: ~XX GB/s
- Minimum data movement: read x once + read γ once + write y once
- For shape (512, 4096): (512 × 4096 × 2 + 4096) × 4 bytes ≈ 16 MB → theoretical minimum = XX μs

## Optimization Versions

### V0: Naive (Single Thread Per Row)

**File:** `rmsnorm_v0_naive.cu`

Each thread processes one entire row: computes sq_sum serially, then normalizes serially.

- **Problem:** No intra-row parallelism, extremely low SM utilization
- **Problem:** Each thread reads the entire row twice (once for sq_sum, once for output)
- **Nsight diagnosis:** Most warps idle, memory bandwidth severely underutilized

### V1: Warp-Level Shuffle Reduction

**File:** `rmsnorm_v1_warp_reduce.cu`

One block per row. Multiple threads cooperatively compute sq_sum using `__shfl_sync` warp reduction.

- **Key insight:** Parallel reduction of sq_sum across threads within a warp
- **Broadcast:** Computed rms is shared via shared memory, then all threads participate in output
- **Limitation:** Only one warp's threads contribute to reduction when blockDim > 32

### V2: Vectorized Memory Access (float4)

**File:** `rmsnorm_v2_vectorized.cu`

- `float4` vectorized loads for sq_sum computation: 128-bit transactions instead of 32-bit
- Cross-warp reduction via shared memory `warp_sums[]` array
- `float4` vectorized stores for output
- All threads participate in both reduction and output writing

### V3: Fused Kernel

**File:** `rmsnorm_v3_fused.cu`

Combines all optimizations into a single fused kernel:

- Vectorized `float4` loads/stores
- Full cross-warp reduction (warp shuffle + shared memory)
- `__fmul_rn` intrinsics for deterministic rounding behavior
- All threads active during both phases (reduction and output)
- Designed for LLM-typical hidden dimensions (4096, 8192)

## Performance Results

<!-- TODO: fill in after running rmsnorm_benchmark_all -->

| Rows | Cols | Naive (ms) | WarpReduce (ms) | Vectorized (ms) | Fused (ms) | cuDNN (ms) |
|------|------|-----------|-----------------|-----------------|-----------|-----------|
| 32 | 32 | | | | | |
| 128 | 256 | | | | | |
| 1111 | 222 | | | | | |
| 4096 | 4096 | | | | | |

## NVIDIA Reference API

- **cuDNN:** `cudnnNormalizationForward` with `CUDNN_NORM_RMS`
