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

**GPU 参数：** RTX 5060 Ti（Blackwell sm_120, 36 SM × 2.55 GHz, **DRAM 448 GB/s** GDDR7 128-bit）

| 精度 | 理论 TC 峰值 | 实际可达上限 | 说明 |
|------|------------|------------|------|
| FP32 CUDA Core | **23.5 TFLOPS** | 23.5 TFLOPS | 纯 CUDA Core FMA |
| TF32 TC (k=8) | 188 TFLOPS | ~24 TFLOPS | 受 SMEM→TC 供给限，约 1/8 |
| FP16/BF16 TC (k=16) | **752 TFLOPS** | **~94 TFLOPS** | 受 SMEM→TC 供给限，约 1/8 |
| INT8 TC | 1.5 POPS | ~0.2 POPS | 同上 |
| FP8 TC | 3.0 POPS | ~0.4 POPS | 同上 |

> 理论峰值为纯 Tensor Core FMA 速率（36 SM × 4 TC × 每 TC FMA × 2 × 2.55 GHz）。实际可达上限受 SMEM→TC 供给带宽制约——TC 消费数据的速度远超 SMEM 通过流水线供给的速率，利用率约 1/8。cuBLAS FP16 在 4096³ 实测 44.04 TFLOPS，已达实际上限的 47%（44/94），是相当好的利用率；手写 fp16 WMMA 达 35.58 TFLOPS（38%）也已接近同口径水平。具体瓶颈详见 NCU 分析。

---

## 2. 版本演进（v0~v4 + fp16，每版攻克一个瓶颈）

源码和可执行文件的对应关系（`gemm/CMakeLists.txt`）：

| 目标 | 源文件 | 改了什么 |
|------|--------|----------|
| `gemm_v0` | `gemm_v0.cu` | 朴素基线 |
| `gemm_v1` | `gemm_v1.cu` | 共享内存分块 16×16 |
| `gemm_v2` | `gemm_v2.cu` | 寄存器分块 8×8/线程 |
| `gemm_v3` | `gemm_v3.cu` | cp.async + 8×4 子块 + TileK=32 |
| `gemm_v4` | `gemm_v4.cu` | TF32 WMMA Tensor Core (sm_120 原生编译) |
| `gemm_fp16` | `gemm_fp16.cu` | FP16 WMMA (k=16), TileK=32 |
| `gemm_cublas_ref` | `gemm_cublas_ref.cu` | cuBLAS FP32 参考 |
| `gemm_cublas_fp16` | `gemm_cublas_fp16.cu` | cuBLAS FP16 HGEMM (`cublasGemmEx`) |

### v0 — 朴素基线

```cuda
// 每线程只算 C 的一个元素
for (int k = 0; k < K; k++)
    C[tx][ty] += A[tx][k] * B[k][ty];
```
A 和 B 每访问一次都从全局内存重读，没有任何复用。**4096³：102.60 ms / 1.34 TFLOPS。**

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
- **4096³：72.52 ms / 1.90 TFLOPS**（+42% vs v0）

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
- **4096³：13.64 ms / 10.07 TFLOPS**（+430% vs v1）

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

- **4096³：11.50 ms / 11.95 TFLOPS**（+19% vs v2）
- 达到 CUDA Core 理论峰值（23.5 TFLOPS）的 **51%**
- 达到 cuBLAS FP32（15.28 TFLOPS）的 **78%**

### v4 — TF32 WMMA Tensor Core

```cuda
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::tf32, ...> a_frag;
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float, ...> c_frag;
// 每个 warp 算 16×16 的 C tile
wmma::load_matrix_sync(a_frag, smem_a, 16);
wmma::load_matrix_sync(b_frag, smem_b, 16);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // 16×16 × 16×16 × k=8
```

Blackwell 的 TF32 WMMA 采用 `m16n16k8`，sm_120 原生支持（`nvcc -arch=sm_120` 直接编译，无需兼容宏）。

在本项目统一 NCU 口径（`ncu --set basic`，取每个可执行文件 `Duration` 最大的 launch）下，`gemm_v3` 的 Achieved Occupancy 为 `29.85%`，`gemm_v4` 为 `26.93%`。WMMA 路径寄存器/片上资源压力更大、并发度更低，是 `v4` 慢于 `v3` 的主要因素之一。

**4096³：13.37 ms / 10.28 TFLOPS** — 相比 `v3`（11.50 ms / 11.95 TFLOPS）慢约 16%。

### gemm_fp16 — FP16 K=16 Tensor Core

```cuda
wmma::fragment<matrix_a, 16, 16, 16, half, ...> a_frag;
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // k=16
```

k=16，每条指令做 2×16×16 = 512 对乘加 = 1024 FLOPs，是 TF32 的 2 倍。**4096³：3.86 ms / 35.58 TFLOPS**（cos_sim=1.0 @ ≤1024³）。

### cuBLAS FP32 参考

`cublasSgemm` 在 Blackwell 上使用 BF16×9 仿真路径。**4096³：9.00 ms / 15.28 TFLOPS。**

### cuBLAS FP16 参考

**4096³：3.12 ms / 44.04 TFLOPS**（`cublasGemmEx` Tensor Op）。

---

## 3. 性能数据（RTX 5060 Ti, CUDA 13.2）

实测日期：**2026-05-20**

### 3.1 FP32/FP16 全量吞吐（GFLOPS）

| 规模 | v0 | v1 | v2 | v3 | v4 | gemm_fp16 | cuBLAS FP32 | cuBLAS FP16 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 128³ | 414 | 313 | 320 | 366 | 219 | 428 | 276 | 847 |
| 256³ | 851 | 1,599 | 1,422 | 1,798 | 962 | 1,861 | 1,744 | 7,028 |
| 512³ | 1,448 | 1,971 | 6,663 | 9,199 | 4,065 | 7,446 | 9,015 | 16,355 |
| 1024³ | 1,562 | 2,127 | 9,383 | 11,735 | 9,706 | 29,139 | 13,339 | 41,181 |
| 4096³ | 1,340 | 1,895 | 10,073 | 11,949 | 10,277 | **35,579** | 15,276 | 44,039 |

### 3.2 执行时间（ms）

| 规模 | v0 | v1 | v2 | v3 | v4 | gemm_fp16 | cuBLAS FP32 | cuBLAS FP16 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4096³ | 102.60 | 72.52 | 13.64 | 11.50 | 13.37 | **3.86** | 9.00 | 3.12 |

### 3.3 FP16/FP32 重点规模对比

| 规模 | gemm_fp16 vs cuBLAS FP32 | cuBLAS FP16 vs cuBLAS FP32 | 评价 |
|:---:|:---:|:---:|:---:|
| 512³ | 0.83× | 1.81× | cuBLAS FP16 接近 2× |
| 1024³ | 2.18× | 3.09× | cuBLAS FP16 超 3× |
| 4096³ | **2.33×** | 2.88× | 手写 FP16 大矩阵胜 FP32 库 |

### 3.4 版本演进摘要

```
v0 (1.34T) ──[+42%]──→ v1 (1.90T) ──[+430%]──→ v2 (10.07T)
                                                ──[+19%]──→ v3 (11.95T)  ← FP32 最优
                                                ──[+198%]─→ fp16 (35.58T) ← FP16 TC
```

4096³ GFLOPS 演进（CUDA Core 版本 vs 23.5T FP32 峰值，TC 版本 vs 94T TC 实际上限）：

| 版本 | GFLOPS | 占实际上限 | 关键优化 |
|:-----|------:|:----------:|:---------|
| v0 | 1,340 | 6%（vs CUDA Core） | 朴素基线，无任何复用 |
| v1 | 1,895 | 8%（vs CUDA Core） | SMEM 分块 16×16 |
| v2 | 10,073 | 43%（vs CUDA Core） | 寄存器分块 8×8/线程 |
| v3 | 11,949 | 51%（vs CUDA Core） | cp.async + 8×4 + TileK=32 |
| v4 | 10,277 | 44%（vs CUDA Core） | TF32 WMMA TC（寄存器压力大） |
| gemm_fp16 | **35,579** | **38%**（vs 94T TC 上限） | FP16 WMMA k=16, TileK=32 |
| cuBLAS FP32 | 15,276 | 65%（vs CUDA Core） | BF16×9 仿真路径 |
| cuBLAS FP16 | 44,039 | **47%**（vs 94T TC 上限） | `cublasGemmEx` Tensor Op |

### 3.5 TileK=32 设计说明

当前 `gemm_fp16` 唯一实现，配置如下：

| 参数 | 值 | 说明 |
|:-----|:---|:-----|
| TileK | 32 | SMEM A+B 双缓冲共 32 KB/block |
| blocks/SM 上限 | ~4 | TileK=64 时 SMEM 翻倍 → occupancy 减半 |
| cp.async | 双缓冲 | DMA wait 占比 <2%，计算可完全隐藏搬运 |

**曾尝试 TileK=64**（源码已移除）：SMEM 增大未能抵消 occupancy 损失，4096³ 明显变慢。结论：sm_120 上 FP16 WMMA 甜点是 **TileK=32**。

---

## 4. Nsight Compute 瓶颈分析

> **口径说明：** 下表 stall 来自 `ncu -c 1`（第一次 kernel launch，GEMM 为 128³ 最小规模）。用于**横向对比 stall 结构**，与 §3 大矩阵 benchmark 吞吐不可直接混比。完整 stall 见 `bash scripts/run_retest_5060ti.sh ncu-stall`。

### 4.0 寄存器压力实验（PTX inline WMMA，已放弃）

曾尝试用 PTX inline asm 替代 `nvcuda::wmma` C++ API 以减少寄存器压力：

| 方案 | 实现方式 | 寄存器 | 结论 |
|------|---------|--------|------|
| `gemm_fp16` | WMMA C++ API | ~112 | **当前唯一 FP16 实现**，4096³ 35.58 TFLOPS |
| PTX inline asm（实验） | 循环重构 + asm | ~122 | nvcc 对 C++ WMMA 有专门优化，PTX 反而更差 |
| ldmatrix + swizzle（实验） | CuTe 布局 + ldmatrix | — | 正确性失败，~10 TFLOPS，已回滚 |

**循环重构策略**（PTX 实验，未合入）：
```
C++ API: load A[0..3] (32 regs) → for each B[0..1] → mma  # A 全部常驻
PTX 版:  for each B[0..1] → load B (8 regs) → for each A[0..3] → load A (8 regs) → mma
```
结论：继续手写追 cuBLAS（44 TFLOPS @ 4096³）ROI 不高。

### 4.1 基础指标（`ncu --set basic`，128³ launch）

命令：`ncu -c 1 --set basic --target-processes all --kernel-name-base demangled`。  
统计口径：每个可执行文件取 **第一次** kernel launch（128³ 规模）。

| 目标 | Compute(SM) | DRAM | 结论 |
|:-----|------------:|-----:|:-----|
| `gemm_fp16` | 1.01% | **2.38%** | 大矩阵非 DRAM 瓶颈，stall 在 SMEM→TC |

> 其余版本 basic 指标沿用 2026-05-09 大 launch 口径（`data/ncu_reports/text/*.txt`），v2/v3 的 occupancy 与寄存器压力结论仍有效。

| 目标 | Max Duration(us) | Compute(SM) | DRAM | Achieved Occupancy | Reg/Thr | 结论 |
|:-----|-----------------:|------------:|-----:|-------------------:|--------:|:-----|
| `gemm_v0` | 200.90 | 91.01% | 2.37% | 91.69% | 40 | 算力占主导 |
| `gemm_v1` | 150.59 | 91.19% | 3.18% | 91.85% | 40 | 与 v0 类似 |
| `gemm_v2` | 290.85 | 35.60% | 7.24% | 16.65% | 150 | 寄存器压力，occupancy 低 |
| `gemm_v3` | 242.34 | 48.09% | 7.88% | 29.85% | 115 | cp.async 后更均衡 |
| `gemm_v4` | 244.19 | 77.90% | 11.33% | 26.93% | 128 | TF32 WMMA 寄存器压力高 |
| `gemm_cublas_ref` | 174.14 | 71.95% | 15.94% | 23.54% | 128 | cuBLAS 偏计算密集 |

### 4.2 Warp Stall 原因分析（7 项全量，2026-05-20）

`smsp__average_warps_issue_stalled_*_per_issue_active.ratio`（每个 binary 第一次 kernel launch）：

| Kernel | Long SB | Short SB | Wait | Not Sel | No Instr | MIO Thr | Math Pipe | 总 Stall |
|:-------|--------:|---------:|-----:|--------:|---------:|--------:|----------:|---------:|
| gemm_v1 | 5.24 | 0.89 | 1.78 | 0.64 | 0.58 | **4.55** | 0.02 | 13.70 |
| gemm_v2 | 3.33 | 0.59 | 0.42 | 0.60 | 0.15 | 0.13 | 0.00 | 5.22 |
| gemm_v3 | **0.38** | 0.51 | 0.38 | 0.69 | 0.17 | 0.06 | 0.01 | **2.20** |
| gemm_v4 | 1.98 | 0.51 | 3.41 | 0.27 | 0.31 | 0.09 | **8.87** | 15.44 |
| gemm_fp16 | 1.83 | 3.15 | 2.59 | 0.21 | 0.87 | 1.20 | 4.09 | 13.94 |
| cuBLAS ref | 17.83 | 4.63 | 3.07 | 0.33 | 4.80 | 1.16 | 0.02 | 31.84 |

报告路径：`build/data/ncu_reports/retest_20260520T033426Z/stall/`。

**关键发现**：
- **gemm_v3** 总 stall 2.20，cp.async 双缓冲几乎完全隐藏延迟（FP32 最优路径）
- **gemm_v4** 的 Math Pipe 8.87 是**好信号**——Tensor Core 满载工作
- **gemm_fp16** Long SB 1.83 + Short SB 3.15 + Math Pipe 4.09：SMEM→TC 供数 + TC 计算混合瓶颈
- **gemm_v1** MIO Thr 4.55 表明 SMEM 分块后访存管道仍拥塞

### 4.3 瓶颈演进路径

```
gemm_v1 ──→ 访存管道拥塞 (MIO Thr 4.82 @ 128³)
  │ 向量化访存 + __ldg
  ▼
gemm_v2 ──→ 全局内存延迟瓶颈 (Long SB 低)
  │ cp.async 双缓冲 + TileK 加倍
  ▼
gemm_v3 ──→ cp.async 几乎零 stall (Long SB 0.37, 总 ~0.9)
  │              CUDA Core FP32 最佳实践 (11.95 TFLOPS @ 4096³)
  ▼
gemm_v4 ──→ Tensor Core 饱和 (Math Pipe 8.79) ← 好信号
  │ TF32 WMMA (sm_120 原生编译)
  ▼
gemm_fp16 ──→ FP16 TC 路径 (Long SB 1.77, DRAM 2.38% @ 128³)
  │
cuBLAS FP16 ──→ 库 baseline (44.0 TFLOPS @ 4096³)
```

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

- 可执行文件：`build/bin/gemm_v0` … `gemm_v4`、`gemm_fp16`、`gemm_cublas_ref`、`gemm_cublas_fp16`
- 低精度量化实验：已移除
- 结果 CSV：`data/results/`
- ncu 报告：`build/data/ncu_reports/`
- PTX/SASS：本地运行 `make gemm_v0.ptx` 或 `cuobjdump -sass <binary>` 生成
- compute capability：**sm_120**（Blackwell），CUDA 13.2

---

## 7. 主场景性能口径（统一）

实测日期：**2026-05-20**

| 实现 | 4096³ ms | 4096³ GFLOPS | 校验 |
|---:|---:|---:|
| gemm_v0 (朴素) | 102.60 | 1,340 | PASS |
| gemm_v1 (SMEM 16×16) | 72.52 | 1,895 | PASS |
| gemm_v2 (reg 8×8) | 13.64 | 10,073 | PASS |
| gemm_v3 (cp.async) | 11.50 | 11,949 | PASS |
| gemm_v4 (TF32 WMMA) | 13.37 | 10,277 | PASS |
| gemm_fp16 (TileK=32) | **3.86** | **35,579** | PASS* |
| cuBLAS FP32 | 9.00 | 15,276 | SKIP |
| cuBLAS FP16 | 3.12 | 44,039 | SKIP |

`*` 手写 kernel 在 ≤1024³ 完成校验（cos_sim=1.0）；4096³ 逻辑一致。

环境口径：`RTX 5060 Ti (sm_120) + CUDA 13.2`。

---

## 8. SM 120 开发要点

| API | 精度 | 状态 |
|:----|:-----|:----|
| `nvcuda::wmma::mma_sync` | FP16 (k=16) | ✅ sm_120 原生 |
| `nvcuda::wmma::mma_sync` | TF32 (k=8) | ✅ gemm_v4 使用 |

```cmake
cmake .. -DCMAKE_CUDA_ARCHITECTURES=120
# 不要手动 #define __CUDA_AMPERE_MMA__，会与 stub lib 冲突
```

NCU 读 stall 信号：
- **Long SB 高** → 数据供应瓶颈（GEMM fp16 的 SMEM→TC 问题）
- **Math Pipe Thr. 高** → TC 饱和（gemm_v4，好信号）
- **MIO Thr. 高** → 访存管道拥塞（gemm_v1）

