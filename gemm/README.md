# GEMM CUDA 优化复盘

本文档记录 `gemm/` 下各版本 kernel 的设计演进。每版只改一个瓶颈。

---

## 1. 数学定义

```
C[m, n] = Σ_{k=0}^{K-1} A[m, k] * B[k, n]
A ∈ R^{M×K}, B ∈ R^{K×N}, C ∈ R^{M×N}
```

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
| `gemm_int8` | `gemm_int8.cu` | INT8 WMMA Tensor Core (k=16) |
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
wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, ...> a_frag;
wmma::fragment<wmma::accumulator, 16, 16, 8, float, ...> c_frag;
// 每个 warp 算 64×32 的 C tile（4×2 个 16×16 fragment）
wmma::load_matrix_sync(a_frag, smem_a, ld);   // SMEM → fragment
wmma::load_matrix_sync(b_frag, smem_b, ld);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // Tensor Core 计算
```

**性能：4096³ — 12.57 ms / 10.94 TFLOPS，比 v3 还慢 14%。** TF32 名义峰值是 CUDA Core 的 8×（188 TFLOPS），实测利用率只有 **5.8%**。

为什么"上 Tensor Core 反而更慢"？逐项拆给最大头：

**A. Occupancy 80%+ → 26.9%（最主要）**

每 warp 8 个 16×16 accumulator fragment：

```cuda
wmma::fragment<wmma::accumulator, 16, 16, 8, float>
    c_frag[kWarpTilesM][kWarpTilesN];   // 4 × 2 = 8 个
```

- 仅累加器 = **64 regs/thread**；加上 a/b fragment + 索引地址 ≈ **128 regs/thread**。
- v3 同样 128 regs/thread，但 8×4 寄存器分块只占 32 regs，其余预算用来隐藏 SMEM 延迟，ncu 给 **80%+ occupancy**。
- v4 因为 fragment 把寄存器吃满，**SM 上只能驻留 1 个 block**，warp scheduler 没有别的 warp 可切——每次 SMEM 加载和 `mma_sync` 都是真等待。

**B. TileK=16 vs v3 的 32 → 外循环次数翻倍**

```cuda
constexpr int kTileK = 16;   // v4
```

- 4096³：v3 外循环 128 轮，v4 **256 轮**，`__pipeline_commit` + `__pipeline_wait_prior` + `__syncthreads()` 全部翻倍。
- 想抬到 32 → 双缓冲 SMEM 32KB→64KB，又要 `cudaFuncSetAttribute` 抬上限，且占用率进一步降低。原 v4 试过这条路，反而慢 13%。

**C. TF32 m16n16k8 的"算密度"只有 FP16 m16n16k16 的 1/3**

| 指令 | SMEM 加载 | 算力 | FLOPs/Byte |
|---|---|---|---|
| `mma.m16n16k8` (TF32) | 1.5 KB | 4096 FLOPs | 2.67 |
| `mma.m16n16k16` (FP16) | 1.0 KB | 8192 FLOPs | **8.0** |

同样 SMEM 带宽，FP16 拿到 3× 算力——这是 `gemm_fp16` 跑到 37.4 TFLOPS、而 v4 卡在 11 TFLOPS 的根本原因之一。**WMMA 选 TF32 这条路本身就有天花板。**

**D. `load_matrix_sync` 是 warp 同步阻塞调用，无法和 mma 重叠**

- v3 的内循环纯 FFMA，可以与 `cp.async` 形成跨 K-tile 的双缓冲流水。
- v4 的 `load_matrix_sync` 把 "SMEM → fragment 内部格式" 的转换塞进同一条指令同步完成，与 `mma_sync` 之间无法跨指令流水。

**E. fragment 复用不充分**

```cuda
for (int kk = 0; kk < kTileK; kk += 8) {
    a_frag[4]; for(i)load a;             // a 复用 j 循环 2 次
    for (int j = 0; j < 2; ++j) {
        b_frag; load b;                   // b 每次 j 重新加载，不复用
        for(i) mma(c_frag[i][j], a, b);
    }
}
```

- a_frag 数组在外，复用 2 次；b 没缓存，每 j 重载——每 kk 步 4+2=6 加载 / 8 mma。
- 把 a/b 都缓存为数组（2×2 fragment 全 cache）能压到 4 加载 / 4 mma。

**一句话总结：** v4 拿了 8× 的名义算力，但被 (A) 累加器吃掉占用率（80%→27%）、(B) TileK 减半导致同步翻倍、(C) TF32 单指令算密度只有 FP16 的 1/3 三件事联合击穿，净结果比 v3 慢 14%。

### gemm_fp16 — FP16 K=16 Tensor Core

```cuda
wmma::fragment<matrix_a, 16, 16, 16, half, ...> a_frag;
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // k=16
```

k=16，每条指令做 2×16×16 = 512 对乘加 = 1024 FLOPs，是 TF32 的 2 倍。数据量减半又让 SMEM 能放 2× 的 K-tile。**4096³：3.68 ms / 37.39 TFLOPS。**

精度（FP32 CPU 参考 vs FP16 GPU 结果，1024³）：

| CosSim | SNR(dB) | MeanAbsErr | MaxAbsErr |
|--------|---------|------------|-----------|
| 1.000000 | 103.63 | 0.00149 | 0.00455 |

FP16 的 10-bit mantissa 对 [-1,1] 分布几乎无损，SNR 比 INT8（45dB）高 58dB。

### cuBLAS FP32 参考

`cublasSgemm` 内部在 Blackwell 上走的是 **BF16×9 仿真**：把 FP32 矩阵拆成 9 个 BF16 子矩阵乘，用 BF16 Tensor Core（k=16）加速，9 次结果重组合回 FP32。**4096³：8.42 ms / 16.33 TFLOPS。**

### gemm_int8 — INT8 WMMA Tensor Core + Per-Channel 量化

kernel 本身和 gemm_fp16 一样走 WMMA (k=16)，只是输入换成了 `int8_t`（signed char），累加器用 `int32_t` 避免溢出。

#### 量化方案对比

对 INT8 推理而言，量化精度和计算性能同等重要。实现了 3 种方案并对比精度：

| 方案 | A (激活) 量化 | B (权重) 量化 | 反量化 |
|------|-------------|-------------|-------|
| Per-Tensor | 一个 `scale_a` 覆盖整个 A | 一个 `scale_b` 覆盖整个 B | `C_deq = C_int32 × scale_a × scale_b` |
| Per-Channel | 一个 `scale_a` 覆盖整个 A | 每列（输出通道）独立 `scale_b[c]` | `C_deq[r][c] = C_int32[r][c] × scale_a × scale_b[c]` |

per-channel 对权重每列独立缩放，保留了通道间的分布差异。实际模型中各通道权重幅值可能差几十倍（如 embedding 层），per-channel 能显著降低量化误差。

#### 精度指标

采用 6 个指标评估 FP32 参考和 INT8 方案之间的差异：

- **余弦相似度** `cos_sim = Σ(r×t) / √(Σr² × Σt²)` — 越接近 1 越好
- **信噪比** `SNR = 10×log₁₀(Σr² / Σ(r-t)²)` (dB) — 越高越好
- **最大相对误差** `max|r-t|/|r|` — 过滤掉接近零的参考值
- **平均绝对误差** `mean|r-t|`
- **P99 绝对误差** — 排除了 1% 的极端值
- **最大绝对误差** `max|r-t|`

#### 精度结果（均匀随机分布 [-1,1]，矩阵 1024³）

| 方案 | CosSim | SNR(dB) | MeanAbsErr | MaxAbsErr |
|------|--------|---------|------------|-----------|
| FP16 WMMA (k=16) | 1.000000 | 103.63 | 0.00149 | 0.00455 |
| Per-Tensor INT8 | 0.999985 | 45.10 | 0.04738 | 0.29195 |
| Per-Channel INT8 | 0.999985 | 45.10 | 0.04733 | 0.27508 |

FP16 比 INT8 高约 **58dB SNR**，平均绝对误差小 **32 倍**（0.0015 vs 0.047）。FP16 的量化步长（2⁻¹⁰ ≈ 0.001）远小于 INT8（1/127 ≈ 0.0079），对 [-1,1] 范围的均匀分布，精度几乎无损。per-channel 在这个均匀分布场景下和 per-tensor 接近，实际模型中权重通道分布差异大时 per-channel 优势更明显。

#### 性能

INT8 在 Tensor Core 上走 k=16，吞吐远高于 FP32 CUDA Core 版本：

| 规模 | 耗时 (ms) | GFLOPS |
|------|-----------|--------|
| 128³ | 0.010 | 429 |
| 256³ | 0.012 | 2915 |
| 512³ | 0.026 | 10318 |
| 1024³ | 0.054 | 40033 |
| 4096³ | 2.107 | 65226 |

4096³ 达到 **65.2 TFLOPS**，接近 FP16 版本的 37.39 TFLOPS 的 1.74×（INT8 数据量减半，带宽压力更小）。

---

## 3. 性能数据（RTX 5060 Ti, CUDA 13.2）

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

### 3.3 相对 cuBLAS FP32 的吞吐比值

| 规模 | v0 | v1 | v2 | v3 | v4 | gemm_fp16 |
|------|----|----|----|----|----|-----------|
| 128³ | 1.89x | 3.19x | 1.24x | 1.53x | 0.99x | 1.50x |
| 256³ | 0.52x | 0.67x | 0.52x | 0.97x | 0.43x | 1.11x |
| 512³ | 0.17x | 0.23x | 0.56x | 1.11x | 0.48x | 1.22x |
| 1024³ | 0.11x | 0.15x | 0.71x | 0.86x | 0.70x | 2.17x |
| 4096³ | 0.09x | 0.12x | 0.49x | 0.77x | 0.67x | 2.29x |

---

## 4. Nsight Compute 瓶颈分析

使用 `ncu --set basic`（4096³ 规模）：

| 版本 | Memory Throughput | DRAM Throughput | Compute Throughput | Occupancy | 瓶颈 |
|:----|:-----------------:|:---------------:|:------------------:|:---------:|:-----|
| v0 朴素 | 93.85% | 41.54% | 93.85% | 高 | **LSU 受限**：每元素 4+ 条全局加载指令，L1 cache 94.22% |
| v3 最优 | 60.58% | 51.30% | 70.37% | 80%+ | **均衡**：cp.async 有效隐藏延迟 |
| v4 WMMA | 37.04% | 37.04% | 93.48% | 26.9% | **Occupancy 崩了**：64 regs/warp 的 fragment 压力，SM 等寄存器就绪 |

对比可以看出：
- v0 的瓶颈不是带宽，而是指令瓶颈——每访存 LSU 都被占满，但利用率低（DRAM 只用 41.54%）
- v3 的 cp.async 把加载卸载到 DMA 单元，释放了 LSU 给计算
- v4 的 WMMA 虽然计算跑满了（Compute 93.48%），但 occupancy 低导致整体不饱和

---

## 5. PTX/SASS 关键指令

所有 kernel 的 PTX 和 SASS 在 `gemm/asm/` 下。

| 版本 | PTX 关键指令 |
|------|-------------|
| v3 | `cp.async.ca.shared.global.L128` — DMA 异步拷贝，不占用 LSU |
| v4 | `wmma.mma.sync.aligned.row.row.m16n16k8.f32.tf32.tf32.f32` |
| gemm_fp16 | `wmma.mma.sync.aligned.row.row.m16n16k16.f32.f16.f16.f32` — k=16 |

---

## 6. 产物路径

- 可执行文件：`build/bin/gemm_v0` … `gemm_v4`、`gemm_fp16`、`gemm_cublas_ref`
- 结果 CSV：`data/results/`
- ncu 报告：`build/data/ncu_reports/`
- PTX/SASS：`gemm/asm/ptx/`、`gemm/asm/sass/`
- compute capability：**sm_120**（Blackwell），CUDA 13.2

## Nsight Compute 性能分析


使用 `ncu --set basic` 对每个可执行文件的第一个 kernel launch 进行 profiling。
运行环境：NVIDIA RTX 5060 Ti (Blackwell sm_120) | CUDA 13.2 | Nsight Compute 2026.1.1

| 版本 | Kernel | Duration(us) | Compute% | MemBW% | L1% | L2% | Occupancy% | Reg/Thread | Block | Grid |
|---|---|---|---|---|---|---|---|---|---|---|
| gemm_v0 | GemmNaiveKernel | 8.4 | 34.1% | 34.1% | 43.6% | 8.2% | 29.4% | 32 | 256 | 64 |
| gemm_v1 | GemmSmemKernel | 6.0 | 36.0% | 36.0% | 43.9% | 7.9% | 28.8% | 40 | 256 | 64 |
| gemm_v2 | GemmRegTiledKernel | 21.2 | 1.0% | 1.9% | 53.3% | 1.2% | 16.6% | 142 | 256 | 1 |
| gemm_v3 | GemmCpAsyncKernel | 9.5 | 2.4% | 4.9% | 43.9% | 2.8% | 16.6% | 128 | 256 | 2 |
| gemm_fp16 | GemmFP16Kernel | 8.7 | 1.0% | 2.1% | 57.5% | 1.4% | 16.6% | 112 | 256 | 1 |
| gemm_fp8 | cublasLtMatmul (cuBLASLt FP8) | 3.0 | 1.6% | 4.0% | 12.1% | 3.1% | 8.3% | 255 | 128 | 16 |
| gemm_int8 | GemmINT8Kernel | 6.6 | 0.7% | 5.5% | 61.0% | 1.4% | 16.5% | 120 | 256 | 1 |
| gemm_cublas_ref | cublasSgemm (cuBLAS FP32) | 5.0 | 7.6% | 8.4% | 54.3% | 5.2% | 8.3% | 80 | 128 | 8 |
**说明：** ncu `--set basic` 默认对程序的**第一个 kernel launch** 进行 profiling。对于 GEMM 等算子，这对应最小测试尺寸（128×128），GPU 远未饱和。因此表格中的 Compute% / MemBW% 表示的是**小尺寸下的资源利用率**，用于横向对比各版本的寄存器压力、occupancy 等结构性差异。大尺寸下的实际性能请参考各算子 README 中的完整 benchmark 表格。

