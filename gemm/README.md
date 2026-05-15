# GEMM CUDA 优化复盘

本文档记录 `gemm/` 下各版本 kernel 的设计演进。每版只改一个瓶颈。

---

## 1. 数学定义

$$
C_{m,n} = \sum_{k=0}^{K-1} A_{m,k}\,B_{k,n}
$$

纯文本：`C[m,n] = sum_{k=0..K-1}(A[m,k] * B[k,n])`。

$$
A \in \mathbb{R}^{M\times K},\quad B \in \mathbb{R}^{K\times N},\quad C \in \mathbb{R}^{M\times N}
$$

纯文本：`A:(M,K), B:(K,N), C:(M,N)`。

方阵（M=N=K）下：
- 计算量：2×N³ FLOPs（N³ 次乘 + N³ 次加）
- 最小搬运：A(N²) + B(N²) + C(N²) = 3N² 个元素 = 12N² bytes
- 算术强度 = 2N³ / 12N² = N/6 FLOP/Byte

| N | 算术强度 | Ridge Point (23.5T/448G≈52) |
|---|----------|-----------------------------|
| 128 | 21 | 访存受限 |
| 512 | 85 | 均衡 |
| 1024 | 171 | 计算受限 |
| 4096 | 683 | 计算受限 |

**GPU 参数：** RTX 5060 Ti（Blackwell sm_120, 36 SM × 2.55 GHz）
- FP32 CUDA Core：36×128×2.55×2 = **23.5 TFLOPS**
- TF32 TC (k=8)：36×2048×2.55 ≈ **188 TFLOPS**（理论，实际受 occupancy 限）
- FP16/BF16 TC (k=16)：36×4096×2.55 ≈ **376 TFLOPS**
- DRAM：**448 GB/s**（GDDR7 × 128-bit）

实际利用率达不到理论值——指令流中有 load/store、地址计算、同步等非 FMA 指令混入。手写 FP32 kernel 通常达到理论峰值的 40~60%。

---

## 2. 版本演进（5 个版本，一个瓶颈一版）

源码和可执行文件的对应关系（`gemm/CMakeLists.txt`）：

| 目标 | 源文件 | 改了什么 |
|------|--------|----------|
| `gemm_v0` | `gemm_v0.cu` | 朴素基线 |
| `gemm_v1` | `gemm_v1.cu` | 共享内存分块 16×16 |
| `gemm_v2` | `gemm_v2.cu` | 寄存器分块 8×8/线程 |
| `gemm_v3` | `gemm_v3.cu` | cp.async + 8×4 子块 + TileK=32 |
| `gemm_v4` | `gemm_v4.cu` | TF32 WMMA Tensor Core |
| `gemm_fp16` | `gemm_fp16.cu` | FP16 WMMA (k=16) |
| `gemm_int8` | `gemm_int8.cu` | INT8 WMMA (k=16)，per-tensor 量化 |
| `quant_gemm_compare` | `quant_gemm_compare.cu` | FP16/FP8/INT8 舍入与 CPU 代理 GEMM 数值对比（无 FP8 kernel） |
| `gemm_fp8_cublaslt` | `gemm_fp8_cublaslt.cu` | cuBLASLt FP8 E4M3→FP32（TN 布局），测库 FP8 吞吐 |
| `gemm_cublas_ref` | `gemm_cublas_ref.cu` | cuBLAS FP32 参考 |

### v0 — 朴素基线

```cuda
// 每线程只算 C 的一个元素
for (int k = 0; k < K; k++)
    C[tx][ty] += A[tx][k] * B[k][ty];
```
A 和 B 每访问一次都从全局内存重读，没有任何复用。**4096³：97.12 ms / 1.42 TFLOPS。**

### v1 — 共享内存分块

将 A 和 B 按 16×16 tile 搬进 SMEM，block 内复用。

```cuda
__shared__ float As[16][16], Bs[16][16];
for (int kk = 0; kk < K; kk += 16) {
    // 协作加载一个 tile → SMEM
    As[tx][ty] = A[row][kk + ty];
    Bs[tx][ty] = B[kk + tx][col];
    __syncthreads();
    for (int k = 0; k < 16; k++)  // SMEM 内累加
        Cval += As[tx][k] * Bs[k][ty];
    __syncthreads();
}
```
- `float4` + `__ldg` 减轻全局加载压力
- **4096³：70.65 ms / 1.95 TFLOPS**（+37% vs v0）

### v2 — 寄存器分块

每线程算 8×8 = 64 个 C 元素。16×16 线程 → CTA 覆盖 128×128 的大块。

```
// 每线程 8×8 的寄存器累加器
float C[8][8] = {0};
for (int kk = 0; kk < K; kk += 16) {
    // 加载 A_tile(8×16), B_tile(16×8) 到寄存器
    float A_reg[8], B_reg[8];
    // 8×16 × 16×8 → 8×8，做 8×8×16 = 1024 次 FMA
    for (int k = 0; k < 16; k++) {
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                C[i][j] += A_reg[i] * B_reg[j];
    }
}
```
- TileK=16，外循环 K/16 轮
- **4096³：17.04 ms / 8.07 TFLOPS**（+314% vs v1）

### v3 — cp.async + 8×4 + TileK=32（FP32 最优）

三个改动都在压榨利用率：

```
1) 8×4 子块：寄存器 32 个（vs v2 的 64）→ occupancy 更高
   8 行 × 4 列 = 32 个 C 的累加器，每个线程算 32 个输出元素

2) TileK=16→32：外循环 K/32 = 128 轮（vs 256），
   __syncthreads 次数减半

3) cp.async 流水线：DMA 搬运下一轮 SMEM tile 的同时，
   当前轮正在计算，不占用 LSU
```

```
SMEM 排布（extern __shared__，48KB）：
  As[0..1][128][32]  ping-pong × 2 = 16384 floats = 64KB? 
  等一下：
  As_double[2][128][32] = 8192 floats = 32KB
  Bs_double[2][32][64]  = 4096 floats = 16KB
  合计 48KB ✓
```

```
cp.async 流水线（简化伪码）：
  // 加载第 0 轮
  cp.async(As[0], A_global, tile_size);
  cp.async(Bs[0], B_global, tile_size);
  __pipeline_commit(); __pipeline_wait_prior(1);
  
  for (kk = 0; kk < K; kk += 32) {
      // 预取下一轮
      cp.async(As[轮次^1], A_global + 下一块, tile_size);
      cp.async(Bs[轮次^1], B_global + 下一块, tile_size);
      __pipeline_commit();
      
      // 计算当前轮，DMA 在后台搬运
      8 行 × 4 列的 FMA 循环:
        for (k = 0; k < 32; k++)
          C_reg[i][j] += As[轮次][i][k] * Bs[轮次][k][j];
      
      __pipeline_wait_prior(1);  // 等搬运完成
  }
```

- **4096³：10.98 ms / 12.52 TFLOPS（+55% vs v2）**
- 达到 CUDA Core 理论峰值（23.5 TFLOPS）的 **53%**
- 达到 cuBLAS FP32（16.33 TFLOPS）的 **77%**

### v4 — TF32 WMMA Tensor Core

```cuda
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::tf32, ...> a_frag;
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float, ...> c_frag;
// 每个 warp 算 16×16 的 C tile
wmma::load_matrix_sync(a_frag, smem_a, 16);
wmma::load_matrix_sync(b_frag, smem_b, 16);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // 16×16 × 16×16 × k=8
```

Blackwell 的 TF32 WMMA 在当前路径下采用 `m16n16k8`。按 warp 级口径，一条 `mma_sync` 的计算量为 `2*m*n*k = 2*16*16*8 = 4096 FLOPs`。对比 CUDA Core FP32 路径，理论上每 SM 每周期可提供 `128 core * 2 FLOPs = 256 FLOPs/cycle/SM`。

在本项目统一 NCU 口径（`ncu --set basic`，取每个可执行文件 `Duration` 最大的 launch）下，`gemm_v3` 的 Achieved Occupancy 为 `29.85%`，`gemm_v4` 为 `26.93%`。WMMA 路径寄存器/片上资源压力更大、并发度更低，是 `v4` 慢于 `v3` 的主要因素之一。

**4096³：12.57 ms / 10.94 TFLOPS** — 相比 `v3`（10.98 ms / 12.52 TFLOPS）慢约 14%。

### gemm_fp16 — FP16 K=16 Tensor Core

```cuda
wmma::fragment<matrix_a, 16, 16, 16, half, ...> a_frag;
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // k=16
```

k=16，每条指令做 2×16×16 = 512 对乘加 = 1024 FLOPs，是 TF32 的 2 倍。数据量减半又让 SMEM 能放 2× 的 K-tile。**4096³：3.68 ms / 37.39 TFLOPS。**

### cuBLAS FP32 参考

`cublasSgemm` 内部在 Blackwell 上走的是 **BF16×9 仿真**：把 FP32 矩阵拆成 9 个 BF16 子矩阵乘，用 BF16 Tensor Core（k=16）加速，9 次结果重组合回 FP32。**4096³：8.42 ms / 16.33 TFLOPS。**

---

## 3. 性能数据（RTX 5060 Ti, CUDA 13.2）

说明：本节为历史批次对比数据；当前发布口径请以文末“主场景性能口径（统一）”和对应 `data/results/*.csv` 为准。

### 3.1 执行时间（ms）

| 规模 | v0 | v1 | v2 | v3 | v4 | cuBLAS FP32 | gemm_fp16 |
|------|------|------|------|------|------|-------------|-----------|
| 128³ | 0.0111 | 0.0066 | 0.0168 | 0.0136 | 0.0211 | 0.0209 | 0.0139 |
| 256³ | 0.0314 | 0.0246 | 0.0313 | 0.0170 | 0.0384 | 0.0164 | 0.0147 |
| 512³ | 0.1861 | 0.1383 | 0.0580 | 0.0292 | 0.0674 | 0.0324 | 0.0266 |
| 1024³ | 1.3743 | 1.0209 | 0.2185 | 0.1803 | 0.2198 | 0.1547 | 0.0714 |
| 4096³ | 97.1207 | 70.6531 | 17.0384 | 10.9755 | 12.5675 | 8.4151 | 3.6755 |

### 3.2 吞吐（GFLOPS）

| 规模 | v0 | v1 | v2 | v3 | v4 | cuBLAS FP32 | gemm_fp16 |
|------|------|------|------|------|------|-------------|-----------|
| 128³ | 379 | 639 | 249 | 307 | 199 | 200 | 301 |
| 256³ | 1068 | 1366 | 1071 | 1977 | 874 | 2048 | 2281 |
| 512³ | 1443 | 1942 | 4625 | 9205 | 3981 | 8275 | 10080 |
| 1024³ | 1563 | 2104 | 9827 | 11909 | 9770 | 13880 | 30069 |
| 4096³ | 1415 | 1945 | 8066 | 12522 | 10936 | 16333 | 37393 |

4096³ 下：
- V3 达到 **53%** 的 FP32 CUDA Core 理论峰值（12.52 / 23.5）
- cuBLAS FP32（16.33 TFLOPS）走 BF16×9 仿真，不受 CUDA Core 上限约束，等效利用 TC k=16 的能力
- gemm_fp16（37.39 TFLOPS）接近 cuBLAS FP16（49.30 TFLOPS）的 **76%**

### 3.2.1 4096³ 重点对照（补充 cuBLAS FP16）

| 实现 | GPU ms | TFLOPS |
|------|-------:|-------:|
| `gemm_v3` | 10.98 | 12.52 |
| `gemm_v4` | 12.57 | 10.94 |
| `cuBLAS FP32` | 8.42 | 16.33 |
| `gemm_fp16` | 3.68 | 37.39 |
| `cuBLAS FP16` | 2.79 | 49.30 |

注：`cuBLAS FP16` 的 `2.79 ms` 由同场景算量 `2MNK` 与 `49.30 TFLOPS` 换算得到。

结论：在 4096³ 主场景，手写 `gemm_fp16` 达到 `cuBLAS FP16` 的约 **76%**（37.39 / 49.30）。

### 3.3 相对 cuBLAS FP32 的吞吐比值

| 规模 | v0 | v1 | v2 | v3 | v4 | gemm_fp16 |
|------|----|----|----|----|----|-----------|
| 128³ | 1.89x | 3.19x | 1.24x | 1.53x | 0.99x | 1.50x |
| 256³ | 0.52x | 0.67x | 0.52x | 0.97x | 0.43x | 1.11x |
| 512³ | 0.17x | 0.23x | 0.56x | 1.11x | 0.48x | 1.22x |
| 1024³ | 0.11x | 0.15x | 0.71x | 0.86x | 0.70x | 2.17x |
| 4096³ | 0.09x | 0.12x | 0.49x | 0.77x | 0.67x | 2.29x |

---

## 4. Nsight Compute 瓶颈分析（2026-05-09）

命令：`ncu --set basic --target-processes all --kernel-name-base demangled`。  
统计口径：每个可执行文件取 **Duration 最大** 的一次 kernel launch（见 `data/ncu_reports/text/*.txt` 与 `data/ncu_reports/summary_by_exe.csv`）。

| 目标 | Max Duration(us) | Compute(SM) | DRAM | Memory | Achieved Occupancy | Reg/Thr | 结论 |
|:-----|-----------------:|------------:|-----:|-------:|-------------------:|--------:|:-----|
| `gemm_v0` | 200.90 | 91.01% | 2.37% | 91.01% | 91.69% | 40 | 算力占主导，访存非瓶颈 |
| `gemm_v1` | 150.59 | 91.19% | 3.18% | 91.19% | 91.85% | 40 | 与 v0 类似，算力打满 |
| `gemm_v2` | 290.85 | 35.60% | 7.24% | 49.74% | 16.65% | 150 | 寄存器压力过高，occupancy 降低 |
| `gemm_v3` | 242.34 | 48.09% | 7.88% | 67.04% | 29.85% | 115 | cp.async 后更均衡，仍受 occupancy 约束 |
| `gemm_v4` | 244.19 | 77.90% | 11.33% | 30.35% | 26.93% | 128 | TF32 WMMA 路径寄存器压力更高，并发度偏低 |
| `gemm_fp16` | 84.67 | 54.89% | 13.55% | 72.83% | 29.65% | 118 | Tensor Core 路径下算存较均衡 |
| `gemm_int8` | 52.54 | 43.33% | 9.13% | 66.58% | 29.18% | 106 | INT8 更快但仍非纯带宽瓶颈 |
| `gemm_fp8_cublaslt` | 845.15 | 87.61% | 39.19% | 60.23% | 16.48% | 255 | 库内核算力利用高，寄存器占用大 |
| `gemm_cublas_ref` | 174.14 | 71.95% | 15.94% | 62.23% | 23.54% | 128 | cuBLAS 路径整体更偏计算密集 |

补充：
- `gemm_v4` 当前通过 `__CUDA_AMPERE_MMA__` 兼容宏路径可在本环境完成编译与分析。
- 详细原始报告在 `data/ncu_reports/text/gemm_*.txt`。

---

## Warp Stall 原因分析

| 版本 | #1 Stall | #2 Stall | #3 Stall | #4 Stall | #5 Stall |
|:----|:---------|:---------|:---------|:---------|:---------|
| v0 | Long Scoreboard 79.8% | Not Selected 12.0% | Wait 7.9% | No Instruction 0.2% | Math Pipe Throttle 0.1% |
| v1 | Mio Throttle 43.5% | Long Scoreboard 40.2% | Not Selected 7.6% | Wait 5.6% | Short Scoreboard 2.7% |
| v2 | Long Scoreboard 52.5% | Short Scoreboard 20.5% | Not Selected 16.3% | Wait 6.5% | Mio Throttle 2.2% |
| v3 | Long Scoreboard 46.4% | Not Selected 25.8% | Short Scoreboard 13.1% | Mio Throttle 6.9% | Wait 5.8% |
| v4 | **Math Pipe Throttle 72.9%** | Wait 15.7% | Long Scoreboard 6.5% | Not Selected 2.7% | Short Scoreboard 0.9% |
| fp16 | Long Scoreboard 46.8% | Short Scoreboard 20.2% | Math Pipe Throttle 16.1% | Wait 9.1% | Mio Throttle 6.7% |

v4 的 Math Pipe Throttle 占 72.9%，说明 Tensor Core WMMA 指令已完全占满计算管道，瓶颈在计算而非访存，与 Nsight 分析中 Compute Throughput 低但 occupancy 受限的结论一致。v0-v3 以 Long Scoreboard 为主，符合 memory-bound 预期。

---

## 5. PTX/SASS 关键指令

所有 kernel 的 PTX 和 SASS 可在本地通过 `nvcc --ptx` 或 `--cubin` 生成（`**/asm/` 已从版本控制中排除）。

| 版本 | PTX 关键指令 |
|------|-------------|
| v3 | `cp.async.ca.shared.global.L128` — DMA 异步拷贝，不占用 LSU |
| v4 | `wmma.mma.sync.aligned.row.row.m16n16k8.f32.tf32.tf32.f32` |
| gemm_fp16 | `wmma.mma.sync.aligned.row.row.m16n16k16.f32.f16.f16.f32` — k=16 |

---

## 6. 产物路径

- 可执行文件：`build/bin/gemm_v0` … `gemm_v4`、`gemm_fp16`、`gemm_int8`、`gemm_fp8_cublaslt`、`quant_gemm_compare`、`gemm_cublas_ref`
- 低精度量化实验数据：[quantization_fp16_fp8_int8.md](quantization_fp16_fp8_int8.md)
- 结果 CSV：`data/results/`
- ncu 报告：`build/data/ncu_reports/`
- PTX/SASS：本地运行 `make gemm_v0.ptx` 或 `cuobjdump -sass <binary>` 生成
- compute capability：**sm_120**（Blackwell），CUDA 13.2

---

## 主场景性能口径（统一）

主指标统一为主场景 `gpu_ms`，NCU 吞吐仅用于瓶颈归因。

| 实现 | 主场景维度 | GPU耗时(ms) | 校验状态 | 数据文件 |
|---|---|---:|---|---|
| `gemm_v3` (FP32) | `M=1024,N=1024,K=1024` | 0.196845 | PASS | `data/results/gemm_v3_results.csv` |
| `gemm_fp16` | `M=4096,N=4096,K=4096` | 4.35315 | NOT_RUN* | `data/results/gemm_fp16_results.csv` |

`*` 说明：`gemm_fp16` 在 `M,N,K<=1024` 的对比项已完成误差校验；主场景 `4096^3` 当前仅记录性能数据，未执行同口径正确性校验。

环境口径：`RTX 5060 Ti (sm_120) + CUDA 13.2`。
统一汇总：`data/results/main_scenario_unified.csv`（retest tag: `20260512_manual_retest`）。

## 已知边界与后续补充

- 当前 FP32/FP16 主口径来自不同目标可执行文件，建议后续增加统一批次的并排实测表。
- 建议补充 `M!=N!=K` 的非方阵场景，以覆盖推理服务常见形状分布。

---

## 7. BF16 WMMA Tensor Core + 全变体对比 (2026-05-15)

新增 `gemm_bf16`（BF16 WMMA k=16 Tensor Core）和统一对比工具 `gemm_all_compare`。

### gemm_bf16 — BF16 K=16 Tensor Core

与 gemm_fp16 结构一致，使用 cp.async 双缓冲流水线，128x128 block，2x4 warp 布局。
BF16 的 mantissa 精度（7-bit）低于 FP16（10-bit），在 GEMM 中误差略大于 FP16。

### 全变体统一对比 (`gemm_all_compare`)

将所有手写 kernel、cuBLAS FP32/FP16/BF16 纳入同一可执行文件、同一数据、同一 timing 循环。

| 规模 | gemm_v3 (FP32) | gemm_fp16 (FP16) | gemm_bf16 (BF16) | cuBLAS FP32 | cuBLAS FP16 | cuBLAS BF16 | 最佳 |
|:----|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 128^3 | 0.0103 ms / 405 GFLOPS | 0.0170 ms / 246 GFLOPS | 0.0173 ms / 242 GFLOPS | 0.0338 ms / 124 GFLOPS | **0.0053 ms / 793 GFLOPS** | 0.0187 ms / 224 GFLOPS | cuBLAS FP16 |
| 256^3 | 0.0183 ms / 1835 GFLOPS | **0.0166 ms / 2025 GFLOPS** | 0.0282 ms / 1190 GFLOPS | 0.0386 ms / 869 GFLOPS | 0.0207 ms / 1619 GFLOPS | 0.0233 ms / 1443 GFLOPS | gemm_fp16 |
| 512^3 | 0.0286 ms / 9398 GFLOPS | **0.0232 ms / 11575 GFLOPS** | 0.0290 ms / 9267 GFLOPS | 0.0506 ms / 5305 GFLOPS | 0.0252 ms / 10652 GFLOPS | 0.0268 ms / 10019 GFLOPS | gemm_fp16 |
| 1024^3 | 0.1935 ms / 11098 GFLOPS | 0.0719 ms / 29866 GFLOPS | 0.0851 ms / 25221 GFLOPS | 0.1550 ms / 13852 GFLOPS | 0.0527 ms / 40734 GFLOPS | **0.0526 ms / 40803 GFLOPS** | cuBLAS BF16 |
| 4096^3 | 11.105 ms / 12377 GFLOPS | 3.717 ms / 36973 GFLOPS | 4.705 ms / 29214 GFLOPS | 8.567 ms / 16043 GFLOPS | **2.898 ms / 47420 GFLOPS** | 2.911 ms / 47221 GFLOPS | cuBLAS FP16 |

### 4096^3 重点对照

| 实现 | GPU ms | TFLOPS | vs cuBLAS FP16 |
|:-----|:-----:|:------:|:--------------:|
| gemm_fp16 | 3.72 | 37.0 | 78% |
| gemm_bf16 | 4.70 | 29.2 | 62% |
| cuBLAS FP16 | 2.90 | 47.4 | 100% |
| cuBLAS BF16 | 2.91 | 47.2 | 99.6% |
| gemm_v3 (FP32) | 11.10 | 12.4 | 26% |
| cuBLAS FP32 | 8.57 | 16.0 | 34% |

分析：
- cuBLAS FP16/BF16 在 4096^3 均达到约 47 TFLOPS，接近 Blackwell TC 理论峰值
- 手写 gemm_fp16 达到 cuBLAS FP16 的 78%，说明 WMMA 手写 kernel 能有效利用 Tensor Core
- 手写 gemm_bf16 略低（62%），瓶颈可能是寄存器分配或 memcpy 粒度
- gemm_v3（FP32 CUDA Core）12.4 TFLOPS，达到 cuBLAS FP32（16.0 TFLOPS）的 77%
- 256^3 和 512^3 下手写 kernel 超过 cuBLAS，说明定制 kernel 在中规模有优势

### 编译与运行

```bash
# 仅手写 + cuBLAS（默认）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j4 --target gemm_all_compare
./build/bin/gemm_all_compare
```

结果 CSV：`data/results/gemm_all_compare_results.csv`

### CUTLASS 在 sm_120 上的兼容性

CUTLASS 4.4.2 已安装，但 `device::Gemm` 默认配置对 sm_100/120 的 tile 参数未特化。
当前版本主对比走 cuBLAS，CUTLASS 通道保留为后续扩展。
