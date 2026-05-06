# GEMM CUDA 优化复盘

本文档描述 `gemm/` 下各版本内核的设计与结论；**命令与路径默认 Windows 11 + PowerShell + Visual Studio 多配置**（可执行文件在 `build/bin/Release/`）。顶层构建步骤见仓库根目录 [`README.md`](../README.md)。

---

## 1. 项目目标

本项目不是单纯实现 GEMM，而是构建一条可解释、可复现、可量化的优化路径：

- 从 **v0** 到 **v4** 逐步演进（中间含 **v3** 双缓冲）；
- 每一步都回答三个问题：上一版有什么问题？为什么这一步合理？数据是否支持结论？

### 1.1 源码与可执行文件（`gemm/CMakeLists.txt`）

| 目标 | 源文件 | 说明 |
|------|--------|------|
| `gemm_v0.exe` | `gemm_v0.cu` | 朴素基线，每线程一个输出元素 |
| `gemm_v1.exe` | `gemm_v1.cu` | 共享内存 16×16 tile，`float4` / `__ldg` |
| `gemm_v2.exe` | `gemm_v2.cu` | 线程级寄存器分块，每线程 **8×8** 子块，块覆盖 128×128 |
| `gemm_v3.exe` | `gemm_v3.cu` | 在 v2 几何上增加 **双缓冲** SMEM |
| `gemm_v4.exe` | `gemm_v4.cu` | **主线**：FP32 大 tile（128×128 CTA，`K` tile=32） |
| `gemm_cublas_ref.exe` | `gemm_cublas_ref.cu` | `cublasGemmEx`，FP16 输入 + FP32 累加，Tensor Core |
| `gemm_benchmark_all.exe` | `benchmark_all.cu` + `gemm_benchmark_v2_v3.cuh` | **汇总**：V0–V1 在 `benchmark_all.cu` 内；**V2/V3 设备核与 `gemm_v2.cu`、`gemm_v3.cu` 对齐**（见 `gemm_benchmark_v2_v3.cuh`，改独立 V2/V3 源时请同步更新该头或改为抽取公共 `.cuh`）；对比 **`cublasSgemm`（FP32）**；**不包含** `gemm_v4` 与 `gemm_cublas_ref` |

运行汇总对比：

```powershell
.\build\bin\Release\gemm_benchmark_all.exe
```

单独跑某版本或 cuBLAS Tensor 参考：

```powershell
.\build\bin\Release\gemm_v3.exe
.\build\bin\Release\gemm_cublas_ref.exe
```

---

## 2. 数学与性能预判

### 2.1 数学定义

```
C[m, n] = Σ_{k=0}^{K-1} A[m, k] × B[k, n]
A ∈ R^{M×K}, B ∈ R^{K×N}, C ∈ R^{M×N}
```

### 2.2 算术强度估算（方阵 N×N）

| 指标 | 公式 | 结果 |
|------|------|------|
| 数据搬运 | `(M*K + K*N + M*N) * 4 bytes` | `12N² bytes` |
| 计算量 | `2*M*N*K FLOPs` | `2N³ FLOPs` |
| 算术强度 | `2N³ / 12N²` | `N/6 FLOP/Byte` |

### 2.3 Roofline 判断

- GPU（项目环境记录）：FP32 峰值约 25 TFLOPS，DRAM 带宽约 448 GB/s；ridge point 约 `55 FLOP/Byte`（量级估计，随卡型变化）。

| N | 算术强度 | 初步判断 |
|---|----------|----------|
| 128 | ~21 | 偏访存受限 |
| 512 | ~85 | 偏计算受限 |
| 1024 | ~170 | 计算受限 |
| 4096 | ~683 | 计算受限 |

小规模先看访存与 launch 效率，大规模重点看 FMA 利用率与占用率。

---

## 3. 版本演进

### 3.1 v0：朴素基线

**文件：** `gemm_v0.cu`

- 每线程只算一个 `C` 元素；A/B 被大量重复从全局内存读取。
- 访存与并行度往往主导，算力难以吃满。

**价值：** 正确性下限与后续加速比的参照。

### 3.2 v1：共享内存分块

**文件：** `gemm_v1.cu`

- 16×16 线程块对应的 tile；在 SMEM 中复用 A/B 条带。
- `float4`、`__ldg` 等减轻全局访问与指令条数。

### 3.3 v2：线程级寄存器分块

**文件：** `gemm_v2.cu`

- 每线程计算 **TM×TN = 8×8** 输出子块；16×16 线程 → CTA 覆盖 **128×128**。
- `kTileK = 16`；`As[kBlockM][kTileK]`、`Bs[kTileK][kBlockN]` 与寄存器累加器协同。

### 3.4 v3：双缓冲共享内存

**文件：** `gemm_v3.cu`

- 在 **与 v2 相同的几何**（128×128 CTA，`kTileK=16`，每线程 8×8）上，将 A/B 的 SMEM 拆成 **双缓冲**：`As[2][128][16]`、`Bs[2][16][128]`（逻辑形状，见源码），计算当前缓冲时协作预取下一轮，用计算掩盖访存延迟。
- SMEM 约为单缓冲的 2 倍（约 32KB 量级），仍在常见每块 SMEM 上限内。

### 3.5 v4：大 tile FP32（主线）

**文件：** `gemm_v4.cu`

- **CUDA Core FP32**：CTA 仍覆盖 **128×128**，`kTileK = 32`，提高 K 维算术强度；`__launch_bounds__(256, 2)`，16×16 线程、每线程 **8×8**。
- SMEM：`float As[128][kTileK+1]`、`Bs[128][kTileK+1]`（`+1` 为 padding，缓解 bank conflict），与寄存器 `float4` 预取配合。
- 与 `cublasSgemm` 精度语义一致，便于与 **FP32** 参考对比；与 `gemm_cublas_ref`（FP16 Tensor）对比时需单独说明数据类型与路径差异。

### 3.6 cuBLAS 参考（Tensor Core）

**文件：** `gemm_cublas_ref.cu`

- `cublasGemmEx` + `CUBLAS_TENSOR_OP_MATH`；输入 FP16、输出 FP32 累加。
- 与主线 **FP32 CUDA** 内核对比时，应标明「库侧 Tensor Core」与「自研 FP32」不是同一硬件路径。

---

## 4. 性能数据说明与参考表

### 4.0 主线 FP32 v4

当前仓库中 **`gemm_v4.cu` 为 FP32 大 tile**；其毫秒/GFLOPS **不在**下表「WMMA v4」列中。请在目标 GPU 上运行 `gemm_v4.exe`（及按需扩展 `gemm_benchmark_all`）后自行记录。

### 4.1 表头约定

下表中 **「WMMA v4」列为历史实验配置**（自研 WMMA / Tensor 路径）的实测数字，用于展示相对 v3 与 cuBLAS Tensor 的量级；**与主线 `gemm_v4.cu`（FP32）无关**。

### 4.2 执行时间（ms）

| 规模 | v1 | v2 | v3 | WMMA v4 | cuBLAS Tensor |
|------|------|------|------|-----------|---------------|
| 128³ | 0.0087 | 0.0204 | 0.0209 | 0.0169 | 0.0043 |
| 256³ | 0.0229 | 0.0333 | 0.0350 | 0.0211 | 0.0063 |
| 512³ | 0.1360 | 0.0621 | 0.0638 | 0.0357 | 0.0146 |
| 1024³ | 1.0761 | 0.2289 | 0.1953 | 0.1010 | 0.0618 |
| 4096³ | 69.0281 | 17.2117 | 11.5116 | 6.3925 | 6.7571 |

### 4.3 吞吐量（GFLOPS）

| 规模 | v1 | v2 | v3 | WMMA v4 | cuBLAS Tensor |
|------|------|------|------|-----------|---------------|
| 128³ | 481.2 | 206.1 | 200.4 | 248.3 | **976.0** |
| 256³ | 1466.7 | 1008.9 | 957.4 | 1593.3 | **5293.2** |
| 512³ | 1973.8 | 4320.0 | 4209.9 | 7521.4 | **18400.1** |
| 1024³ | 1995.6 | 9379.8 | 10998.2 | 21253.8 | **34721.1** |
| 4096³ | 1991.1 | 7985.2 | 11939.2 | **21499.9** | 20339.9 |

### 4.4 相对 cuBLAS Tensor 的比值（表中 WMMA v4）

| 规模 | v2/Tensor | v3/Tensor | WMMA v4/Tensor | 说明 |
|------|-----------|-----------|------------------|------|
| 128³ | 0.21× | 0.21× | 0.25× | 中小矩阵库侧通常更快 |
| 256³ | 0.19× | 0.18× | 0.30× | |
| 512³ | 0.23× | 0.23× | 0.41× | |
| 1024³ | 0.27× | 0.32× | 0.61× | |
| 4096³ | 0.47× | 0.71× | **1.06×** | 该实验配置下与 Tensor 参考接近 |

---

## 5. 结果解读（与主线对照）

1. **v2**：128×128 CTA、`kTileK=16`、每线程 8×8；写回可向量化。
2. **v3**：双缓冲在 **大矩阵** 上更易体现「加载与计算重叠」；小矩阵受并行度与同步开销影响，收益可能有限。
3. **主线 v4（FP32）**：通过 **`kTileK=32`** 与 SMEM 组织提高强度；与 v3 的优劣需用 **同一精度、同一规模** 重测对比。
4. **WMMA 实验 v4（表中）**：在 **不同精度/指令路径** 下相对 v3 可大幅提升峰值吞吐；与工业库差距来自调度、多级 tiling、架构专项优化等。
5. **`gemm_benchmark_all`**：对比的是 **V0–V3 + `cublasSgemm`**；要看 Tensor Core 库性能请跑 **`gemm_cublas_ref.exe`**。

---

## 6. 产物路径与工程附注

- **可执行文件（VS 多配置）：** `build/bin/Release/*.exe`（Ninja 常为 `build/bin/`）。
- **结果 CSV：** 各 `main` 写入 `data/results/`（若根 `.gitignore` 忽略 `data/`，则仅本地存在）。
- **自动生成 ISA：** `CMakeLists.txt` 中 PTX/cubin 规则当前示例为 **`arch=compute_89,code=sm_89`**，与文档中 Blackwell **sm_120** 可能不一致；导出真实 ISA 时请改为你的 `CMAKE_CUDA_ARCHITECTURES` 或单独调用 nvcc。

---

## 7. 参考 API

- cuBLAS：`cublasSgemm`、`cublasGemmEx`
- cuBLASLt：`cublasLtMatmul`
- CUTLASS：模板化高性能 GEMM 参考实现
