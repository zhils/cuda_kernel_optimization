# GEMM CUDA 优化复盘

本文档描述 `gemm/` 下各版本内核的设计与结论；**命令与路径默认 Ubuntu 22.04 + CUDA 13.2**（可执行文件在 `build/bin/`）。顶层构建步骤见仓库根目录 [`README.md`](../README.md)。

---

## 1. 项目目标

本项目不是单纯实现 GEMM，而是构建一条可解释、可复现、可量化的优化路径：

- 从 **v0** 到 **v5** 逐步演进；
- 每一步都回答三个问题：上一版有什么问题？为什么这一步合理？数据是否支持结论？

### 1.1 源码与可执行文件（`gemm/CMakeLists.txt`）

| 目标 | 源文件 | 说明 |
|------|--------|------|
| `gemm_v0` | `gemm_v0.cu` | 朴素基线，每线程一个输出元素 |
| `gemm_v1` | `gemm_v1.cu` | 共享内存 16x16 tile，`float4` / `__ldg` |
| `gemm_v2` | `gemm_v2.cu` | 线程级寄存器分块，每线程 **8x8** 子块，块覆盖 128x128 |
| `gemm_v3` | `gemm_v3.cu` | **FP32 最优**：8×4 子块 + TileK=32 + `cp.async` 双缓冲 SMEM |
| `gemm_v4` | `gemm_v4.cu` | **TF32 WMMA** Tensor Core：128×128 块 + 8 warp WMMA |
| `gemm_v5` | `gemm_v5.cu` | **WGMMA** Blackwell Tensor Core：128×128 块 + WGMMA 指令 |
| `gemm_fp16` | `gemm_fp16.cu` | **FP16 WMMA (k=16)**：128×128 块 + half Tensor Core |
| `gemm_cublas_ref` | `gemm_cublas_ref.cu` | `cublasSgemm`（FP32 输入 + FP32 累加）参考实现 |

单独跑某版本或 cuBLAS 参考：

```bash
./build/bin/gemm_v3
./build/bin/gemm_cublas_ref
```

---

## 2. 数学与性能预判

### 2.1 数学定义

```
C[m, n] = Σ_{k=0}^{K-1} A[m, k] * B[k, n]
A ∈ R^{MxK}, B ∈ R^{KxN}, C ∈ R^{MxN}
```

### 2.2 算术强度估算（方阵 NxN）

| 指标 | 公式 | 结果 |
|------|------|------|
| 数据搬运 | `(M*K + K*N + M*N) * 4 bytes` | `12N^2 bytes` |
| 计算量 | `2*M*N*K FLOPs` | `2N^3 FLOPs` |
| 算术强度 | `2N^3 / 12N^2` | `N/6 FLOP/Byte` |

### 2.3 Roofline 判断

- **GPU：** RTX 5060 Ti (sm_120, Blackwell)，FP32 峰值 ~25 TFLOPS，Tensor Core FP16 峰值更高；DRAM 带宽 ~448 GB/s。

| N | 算术强度 | 初步判断 |
|---|----------|----------|
| 128 | ~21 | 偏访存受限 |
| 512 | ~85 | 偏计算受限 |
| 1024 | ~170 | 计算受限 |
| 4096 | ~683 | 计算受限 |

---

## 3. 版本演进

### 3.1 v0：朴素基线

**文件：** `gemm_v0.cu`

- 每线程只算一个 `C` 元素；A/B 被大量重复从全局内存读取。
- **4096³：97.12 ms / 1.42 TFLOPS**

### 3.2 v1：共享内存分块

**文件：** `gemm_v1.cu`

- 16x16 线程块的共享内存 tile；`float4`、`__ldg` 减轻全局访问。
- **4096³：70.65 ms / 1.95 TFLOPS**（+37% vs v0）

### 3.3 v2：线程级寄存器分块

**文件：** `gemm_v2.cu`

- 每线程 8x8 输出子块，16x16 线程 → CTA 覆盖 128x128。
- **4096³：17.04 ms / 8.07 TFLOPS**（+314% vs v1）

### 3.4 v3：cp.async + 8×4 子块 + TileK=32 ← FP32 最优

**文件：** `gemm_v3.cu`

- 8×4 子块（32 寄存器 vs 原 8×8 的 64 寄存器）→ 更高 occupancy
- TileK=16→32 → 外循环减半 → __syncthreads 减少
- `cp.async` DMA 异步加载与计算重叠
- `__launch_bounds__(256,2)` 保证 2 blocks/SM
- **4096³：10.98 ms / 12.52 TFLOPS**（+55% vs v2）

### 3.5 v4：TF32 WMMA Tensor Core

**文件：** `gemm_v4.cu`

- `wmma::mma_sync` 使用 Blackwell TF32 Tensor Core（m16n16k8）
- 128×128 块 + 8 warp 全部参与 WMMA 计算
- **4096³：12.57 ms / 10.94 TFLOPS** — 慢于 V3
- **原因：** Blackwell TF32 仅支持 k=8（每指令 2048 FMA），而 CUDA Core FMA 调度更灵活

### 3.6 v5：WGMMA Blackwell Tensor Core

**文件：** `gemm_v5.cu`

- 使用 Blackwell 的 `wgmma`（warp-group MMA）指令
- 128×128 块 + warp-group 协同
- **4096³：12.21 ms / 11.26 TFLOPS** — 略好于 V4 但仍低于 V3

### 3.7 gemm_fp16：FP16 Tensor Core（k=16）

**文件：** `gemm_fp16.cu`

- half WMMA，m16n16k16（每指令 **4096 FMA**，是 TF32 的 2 倍）
- 128×128 块 + 8 warp WMMA
- **4096³：3.68 ms / 37.39 TFLOPS** — 碾压所有 FP32 方案
- **关键：** FP16 的 k=16 使 Tensor Core 吞吐翻倍，是 Blackwell 上的最佳选择

### 3.8 cuBLAS 参考（FP32）

**文件：** `gemm_cublas_ref.cu`

- `cublasSgemm`（FP32 API），内部使用 BF16x9 仿真算法
- **4096³：8.42 ms / 16.33 TFLOPS**
- cuBLAS 在 Blackwell 上自动使用 BF16 Tensor Core 做 FP32 仿真，是所有手写 FP32 kernel 无法比拟的

---

## 4. 性能数据（Ubuntu 22.04 + RTX 5060 Ti + CUDA 13.2）

### 4.1 执行时间（ms，越短越好）

| 规模 | v0 | v1 | v2 | v3 | v4 | v5 | cuBLAS FP32 | gemm_fp16 |
|------|------|------|------|------|------|------|-------------|-----------|
| 128³ | 0.0111 | **0.0066** | 0.0168 | 0.0136 | 0.0211 | 0.0164 | 0.0209 | 0.0139 |
| 256³ | 0.0314 | 0.0246 | 0.0313 | **0.0170** | 0.0384 | 0.0175 | 0.0164 | 0.0147 |
| 512³ | 0.1861 | 0.1383 | 0.0580 | **0.0292** | 0.0674 | 0.0397 | 0.0324 | 0.0266 |
| 1024³ | 1.3743 | 1.0209 | 0.2185 | 0.1803 | 0.2198 | 0.2175 | **0.1547** | 0.0714 |
| 4096³ | 97.1207 | 70.6531 | 17.0384 | 10.9755 | 12.5675 | 12.2069 | **8.4151** | 3.6755 |

### 4.2 吞吐量（GFLOPS，越大越好）

| 规模 | v0 | v1 | v2 | v3 | v4 | v5 | cuBLAS FP32 | gemm_fp16 |
|------|------|------|------|------|------|------|-------------|-----------|
| 128³ | 378.7 | **638.8** | 249.2 | 307.4 | 198.8 | 255.4 | 200.3 | 300.8 |
| 256³ | 1068.2 | 1365.9 | 1071.0 | 1977.0 | 874.3 | 1913.1 | **2048.4** | 2281.5 |
| 512³ | 1442.6 | 1941.5 | 4625.1 | **9205.1** | 3980.7 | 6759.6 | 8275.2 | 10080.0 |
| 1024³ | 1562.6 | 2103.6 | 9827.3 | 11909.3 | 9769.7 | 9874.6 | **13880.1** | 30069.4 |
| 4096³ | 1415.1 | 1945.3 | 8066.4 | 12522.4 | 10936.0 | 11259.1 | **16332.5** | 37392.8 |

### 4.3 相对 cuBLAS FP32 的吞吐比值

| 规模 | v0 | v1 | v2 | v3 | v4 | v5 | gemm_fp16 |
|------|----|----|----|----|----|----|-----------|
| 128³ | 1.89x | **3.19x** | 1.24x | 1.53x | 0.99x | 1.27x | 1.50x |
| 256³ | 0.52x | 0.67x | 0.52x | **0.97x** | 0.43x | 0.93x | 1.11x |
| 512³ | 0.17x | 0.23x | 0.56x | **1.11x** | 0.48x | 0.82x | 1.22x |
| 1024³ | 0.11x | 0.15x | 0.71x | 0.86x | 0.70x | 0.71x | **2.17x** |
| 4096³ | 0.09x | 0.12x | 0.49x | 0.77x | 0.67x | 0.69x | **2.29x** |

### 4.4 优化路径总结（4096³）

| 版本 | 耗时 | TFLOPS | vs 上一版 | vs cuBLAS |
|:----|:----:|:------:|:---------:|:---------:|
| **v0** 朴素 | 97.12 ms | 1.42 | — | 8.7% |
| **v1** +SMEM | 70.65 ms | 1.95 | +37% | 11.9% |
| **v2** +register tile | 17.04 ms | 8.07 | +314% | 49.4% |
| **v3** +cp.async+T32+8×4 | **10.98 ms** | **12.52** | **+55%** | **76.7%** |
| v4 TF32 WMMA | 12.57 ms | 10.94 | -13% | 67.0% |
| v5 WGMMA | 12.21 ms | 11.26 | -10% | 68.9% |
| **gemm_fp16** half TC | **3.68 ms** | **37.39** | **+264%** | **229%** |
| cuBLAS FP32 | 8.42 ms | 16.33 | — | 100% |

---

## 5. 关键发现

### 5.1 CUDA Core FMA > TF32 WMMA（sm_120）

Blackwell 的 TF32 Tensor Core 只支持 **m16n16k8**（每指令 2048 FMA），而 CUDA Core FMA 调度更灵活。**V3（12.52 TFLOPS）比 V4（10.94 TFLOPS）快 14%。**

### 5.2 WGMMA 未超越 WMMA

V5 的 WGMMA（11.26 TFLOPS）与 V4 的 WMMA（10.94 TFLOPS）性能接近，差距在测量误差范围内。两种 Tensor Core API 在 sm_120 上受相同的 k=8 硬件限制。

### 5.3 FP16 Tensor Core 才是杀手锏

**gemm_fp16（37.39 TFLOPS）比最佳 FP32 手写 kernel（V3，12.52 TFLOPS）快 3×**。原因：
- FP16 Tensor Core 支持 **k=16**（每指令 4096 FMA）
- 数据量减半，同等 SMEM 可存储 2× K-tile
- 达到 cuBLAS FP16（约 49 TFLOPS）的 **76%**

### 5.4 cuBLAS FP32 的 BF16x9 仿真

cuBLAS 在 Blackwell 上对 `cublasSgemm` 内部使用 **BF16 Tensor Core + BF16x9 仿真算法**，将每个 FP32 矩阵乘分解为 9 个 BF16 运算，利用 k=16 的 BF16 Tensor Core 实现净加速。这是手写 FP32 kernel 无法匹敌的根本原因。

---

## 6. 产物路径与工程附注

- **可执行文件：** `build/bin/`
- **结果 CSV：** 各 `main` 写入 `data/results/`
- **CUDA 架构：** RTX 5060 Ti 为 Blackwell 架构，Compute Capability **sm_120**，CUDA 13.2

---

## 7. 参考 API

- cuBLAS：`cublasSgemm`、`cublasGemmEx`
- cuBLASLt：`cublasLtMatmul`
- WMMA：`nvcuda::wmma` 命名空间下的 `load_matrix_sync`、`mma_sync`、`store_matrix_sync`
- WGMMA：Blackwell `wgmma` warp-group MMA 指令
