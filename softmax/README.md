# Softmax

## Mathematical Definition

For each row vector x of length C:

```
softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

Numerically stable formulation subtracts `max(x)` to prevent overflow in `exp()`.

## Theoretical Performance Analysis

### Arithmetic Intensity

| Metric | Formula | Value |
|--------|---------|-------|
| Data movement | Read input + write output = 2 × R × C × 4 bytes | 8RC bytes |
| Computation | R × (C comparisons + C exp + C additions + C divisions) | ~4RC FLOPs |
| Arithmetic intensity | 4RC / 8RC | **0.5 FLOP/Byte** |

### Roofline Classification

With arithmetic intensity of ~0.5 FLOP/Byte, Softmax is deeply **memory bound** regardless of input size.
Optimization should focus on maximizing DRAM bandwidth utilization and minimizing memory passes.

### Theoretical Peak

- RTX 5060 Ti DRAM bandwidth: ~XX GB/s
- Minimum data movement (online softmax): read once + write once = 2 × R × C × 4 bytes
- Theoretical minimum time for 512×4096: (2 × 512 × 4096 × 4) / (XX × 10⁹) = XX μs

## Optimization Versions

### V0: Naive (Single Thread Per Row)

**File:** `softmax_v0_naive.cu`

Each thread processes one entire row sequentially: 3 passes (max → exp+sum → normalize).

- **Problem:** No parallelism within a row — wasted GPU resources for large C
- **Problem:** 3 passes over global memory per row
- **Nsight diagnosis:** Very low SM utilization, memory bandwidth underutilized

### V1: Shared Memory Block Reduction

**File:** `softmax_v1_shared_mem.cu`

One block per row. Threads cooperatively compute max and sum via shared memory tree reduction.

- **Key insight:** Parallel reduction turns O(C) serial work into O(C/T + log T) parallel work
- **Still 3 passes:** max reduction → exp+sum → normalize
- **Shared memory usage:** `blockDim.x × sizeof(float)` for reduction workspace

### V2: Online Softmax (Single-Pass)

**File:** `softmax_v2_online.cu`

Implements the online normalizer algorithm (Milakov & Gimelshein, 2018): maintains running max and sum simultaneously, rescaling partial sums when a new maximum is found.

- **Key insight:** Reduces memory passes from 3 to 2 (one accumulation pass + one output pass)
- **Numerical stability:** Maintained by on-the-fly rescaling: `sum = sum × exp(old_max - new_max) + exp(val - new_max)`
- **Block reduction:** Pairs `(max, sum)` are reduced together with proper rescaling

### V3: Warp Shuffle + Vectorized Loads

**File:** `softmax_v3_warp_shuffle.cu`

For rows that fit within a single warp (cols ≤ 128):

- `__shfl_down_sync` for intra-warp max/sum reduction — **zero shared memory**
- `float4` vectorized loads: 4× fewer memory transactions
- `__shfl_sync(..., 0)` broadcasts final max/sum to all lanes
- Highest occupancy due to zero shared memory usage

### cuDNN Reference

**File:** `softmax_cudnn_ref.cu`

`cudnnSoftmaxForward` with `CUDNN_SOFTMAX_ACCURATE` mode — NVIDIA's optimized implementation.

## Performance Results

<!-- TODO: fill in after running softmax_benchmark_all -->

| Rows | Cols | Naive (ms) | SharedMem (ms) | Online (ms) | WarpShuffle (ms) | cuDNN (ms) |
|------|------|-----------|----------------|------------|------------------|-----------|
| 64 | 512 | | | | | |
| 128 | 1024 | | | | | |
| 256 | 4096 | | | | | |
| 512 | 4096 | | | | | |

## NVIDIA Reference API

- **cuDNN:** `cudnnSoftmaxForward` / `cudnnSoftmaxBackward`
