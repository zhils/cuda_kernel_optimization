# FP16 / FP8 / INT8 量化数据汇总（本项目实测）

本文汇总仓库内与 **FP16、FP8(E4M3)、INT8** 相关的**可复现**精度与性能数据。

## 环境与数据文件

| 项 | 说明 |
|----|------|
| GPU | RTX 5060 Ti，Compute Capability 12.0（与 [`benchmark_environment.md`](benchmark_environment.md) 一致） |
| 本报告实测 | WSL2，CUDA **13.2**，2026-05-09 |
| 性能 CSV | `data/results/gemm_fp16_results.csv`、`data/results/gemm_int8_results.csv`、`data/results/gemm_fp8_cublaslt_results.csv` |
| 数值对比 CSV | `data/results/quant_gemm_numerical.csv`（由 `quant_gemm_compare` 生成） |

复现（仓库根目录）：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target gemm_fp16 gemm_int8 gemm_fp8_cublaslt quant_gemm_compare
./build/bin/gemm_fp16
./build/bin/gemm_int8
./build/bin/gemm_fp8_cublaslt
./build/bin/quant_gemm_compare
```

## 1. 精度

### 1.1 手写 GEMM kernel 相对 FP32 参考（GPU 验算）

与 `gemm_fp16.cu` / `gemm_int8.cu` 一致：矩阵由 `common::InitMatrix` 初始化；方阵 `K=M`；仅当 `M,N ≤ 1024` 时与 CPU FP32 GEMM 对比。阈值：FP16 用 `1e-1`，INT8 用 `1.0`（因含 per-tensor 量化缩放）。

| 规模 | FP16 `max_abs_diff` | INT8 `max_abs_diff` | INT8 `quant_err`（单矩阵元素最大量化误差） |
|------|---------------------|---------------------|------------------------------------------|
| 128³ | 2.23e-3 | 3.83e-2 | 3.93e-3 |
| 256³ | 2.86e-3 | 5.42e-2 | 3.93e-3 |
| 512³ | 3.56e-3 | 6.28e-2 | 3.93e-3 |
| 1024³ | 4.55e-3 | 5.66e-2 | 3.93e-3 |
| 4096³ | （未做 CPU 对比，标记 SKIP） | 同左 | 3.93e-3 |

### 1.2 舍入与「仅量化输入 + FP32 累加」的 GEMM 误差（CPU，`quant_gemm_compare`）

用于在**同一分布**下对比三种存储格式，**不**代表 Tensor Core 内部累加精度。

- **Round-trip**：元素经 FP16 / FP8 E4M3 / INT8（per-tensor 反量化）后的最大绝对误差。
- **GEMM 输出误差**：对 A、B 做上述 round-trip 后，用 **FP32 累加**做朴素 GEMM，与全 FP32 参考 C 的最大绝对差。`4096³` 因 CPU 代价过大跳过（CSV 中为 `SKIP`）。

| 规模 | max round-trip FP16 | max round-trip FP8 E4M3 | max round-trip INT8 反量化 | GEMM 输出 err (FP16 输入) | GEMM 输出 err (FP8 输入) | GEMM 输出 err (INT8 输入) |
|------|---------------------|-------------------------|----------------------------|---------------------------|---------------------------|---------------------------|
| 128³ | 2.42e-4 | 3.10e-2 | 3.93e-3 | 2.22e-3 | 2.62e-1 | 3.83e-2 |
| 256³ | 2.42e-4 | 3.10e-2 | 3.93e-3 | 2.80e-3 | 3.93e-1 | 5.42e-2 |
| 512³ | 2.42e-4 | 3.10e-2 | 3.93e-3 | 3.27e-3 | 4.51e-1 | 6.28e-2 |
| 1024³ | 2.42e-4 | 3.10e-2 | 3.93e-3 | 3.39e-3 | 3.45e-1 | 5.66e-2 |

**解读（本数据生成方式下）**：在 `InitMatrix` 数值范围内，FP16 舍入远小于 INT8 per-tensor，更小於 FP8 E4M3 的输入误差；但 GEMM 会沿 K 维累加放大误差，故输出端 FP8 代理误差显著大于 FP16。真实 FP8 Tensor Core 路径还需结合累加器类型与缩放策略单独评测。

### 1.3 FP8：库内实现与精度说明

- **性能**：使用 **`gemm_fp8_cublaslt`**（`gemm/gemm_fp8_cublaslt.cu`）：**cuBLASLt** Tensor Core 路径，输入 **E4M3**，输出 **FP32**，布局为文档要求的 **TN**（`transA=T`，`transB=N`），与 [NVIDIA LtFp8Matmul 示例](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtFp8Matmul) 一致；缩放因子为设备端标量 `1.0f`（`A_SCALE_POINTER` / `B_SCALE_POINTER`）。
- **尚无手写 FP8 WMMA**：不存在与 `gemm_fp16.cu` 同级的 `gemm_fp8.cu`。
- **CPU 代理精度**：`quant_gemm_compare` 仍可用于对比舍入与「FP32 累加朴素 GEMM」误差，与 cuBLASLt 内核数值不必一致。

## 2. 性能（手写 kernel，`cudaEvent` 平均）

表内为本次环境 CSV；GFLOPS 按 `2MNK/time` 统计（INT8 与 FP16 使用同一公式便于对照，非「整数 OP/s」）。

| 规模 | FP16 ms | FP16 GFLOPS | INT8 ms | INT8 GFLOPS | INT8 / FP16 时间比 |
|------|---------|-------------|---------|-------------|-------------------|
| 128³ | 0.0116 | 361 | 0.0168 | 249 | 1.45× |
| 256³ | 0.0212 | 1583 | 0.0257 | 1307 | 1.21× |
| 512³ | 0.0331 | 8121 | 0.0321 | 8364 | 0.97× |
| 1024³ | 0.0866 | 24806 | 0.0784 | 27377 | 0.91× |
| 4096³ | 5.24 | 26215 | 2.55 | 53890 | 0.49× |

说明：大矩阵上 INT8 WMMA 本实现更快，与「每周期有效标量乘加数」及访存 footprint 更小有关；与 [`gemm/README.md`](../gemm/README.md) 中另一批次计时（如 4096³ FP16 3.68 ms）可能因驱动/功耗/时钟存在偏差，以本机 CSV 为准。

### 2.1 FP8 性能（cuBLASLt，`gemm_fp8_cublaslt`）

与 2 节相同计时方式（warmup 3、重复 10、`cudaEvent` 平均）。GFLOPS 仍按 `2MNK/time` 便于与 FP16/INT8 对照。4096³ 未跑 CPU 参考，表中 `check=SKIP`。

| 规模 | FP8 ms | FP8 GFLOPS | 相对 INT8 时间比（INT8/FP8） | `max_abs_diff`（≤1024 相对 CPU FP32） |
|------|--------|------------|------------------------------|----------------------------------------|
| 128³ | 0.0100 | 418 | 1.68× | 0.26 |
| 256³ | 0.0087 | 3849 | 2.95× | 0.39 |
| 512³ | 0.0185 | 14516 | 1.73× | 0.45 |
| 1024³ | 0.0269 | 79740 | 2.91× | 0.34 |
| 4096³ | 0.845 | **162632** | 3.02× | — |

**解读**：cuBLASLt 在大方阵上远高于当前手写 FP16/INT8 kernel，属于库优化 + FP8 Tensor Core 峰值利用；小矩阵受 launch 与访存比例影响，优势随 N 增大而更明显。

## 3. 相关源码

| 内容 | 路径 |
|------|------|
| FP16 WMMA GEMM | `gemm/gemm_fp16.cu` |
| INT8 WMMA GEMM + per-tensor 量化 | `gemm/gemm_int8.cu` |
| 三种格式数值对比工具 | `gemm/quant_gemm_compare.cu` |
| FP8 E4M3 性能（cuBLASLt） | `gemm/gemm_fp8_cublaslt.cu` |
