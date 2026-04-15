# GEMM — 通用矩阵乘

## 数学定义

```
C[m, n] = Σ_{k=0}^{K-1} A[m, k] × B[k, n]

A ∈ R^{M×K},  B ∈ R^{K×N},  C ∈ R^{M×N}
```

## 理论性能分析

### 算术强度


| 指标    | 公式                          | 数值（方阵 N×N）        |
| ----- | --------------------------- | ----------------- |
| 数据搬运量 | (M×K + K×N + M×N) × 4 bytes | 12N² bytes        |
| 计算量   | 2 × M × N × K FLOPs         | 2N³ FLOPs         |
| 算术强度  | 2N³ / 12N²                  | **N/6 FLOP/Byte** |


### Roofline 分类

RTX 5060 Ti 规格：

- FP32 峰值算力：~25 TFLOPS
- DRAM 带宽：~448 GB/s
- 平衡点（ridge point）：~55 FLOP/Byte


| 矩阵大小（N×N） | 算术强度 | 分类       |
| --------- | ---- | -------- |
| 128       | ~21  | **访存受限** |
| 512       | ~85  | **计算受限** |
| 1024      | ~170 | **计算受限** |
| 4096      | ~683 | **计算受限** |


对于典型深度学习尺寸（N > 256），GEMM 属于 **计算受限**。（硬件实际性能距离理论性能有些许差距，基本上可以认为40+就会到平衡点，从 访存受限 变为 计算受限。
优化重点应放在提升 FMA 利用率并减少 warp stall。

## 优化版本

### V0：朴素基线

**文件：** `gemm_v0_naive.cu`

每个线程计算 C 的一个元素，从全局内存读取 A 的整行和 B 的整列。

- **问题：** 全局内存重复读取严重——每个 A/B 元素会被读取 M 或 N 次。此时计算强度 = 1/2*4=0.125(1次乘加，2次数据读取)，属于 访存受限，远远不及理论上的 计算受限。
- **Nsight 诊断：** warp stall 主要由 `Stall Long Scoreboard`（内存延迟）和 `Stall LG Throttle`（LSU 饱和）主导
  Nsight的性能图可以参考图gemm_v0_naive.png，可以从图中看到stall中占比前3的分别是：
  a. stall LG Throttle:访存拥塞：因内存子系统拥塞，内存指令（LDG）被节流，无法向LSU（Load/Store Unit）发射。
  b. Stall Long Scoreboard: 等待长延迟数据：等待一个离开SM的长延迟操作结果（如全局内存读取），数据还未返回。
  c. Stall Not selected: 等待调度：指令和依赖都已就绪，但当前周期没被调度器选中，在排队等待。
- **优化计划：** 因此，需要优化内存加载，主要从以下三个方面：
  a. 数据读取：
     (i) 减少全局内存的读写；目标为减少到MxK + KxN + MxN；利用寄存器存储、共享内存存储过程变量；
     (ii) 优化全局内存的读写形式；目标为尽可能少的指令加载上一步叙述的数据量；如向量化加载，访存对齐，只读缓存，内存对齐等；
  b. 数据计算：提高计算强度，尽可能让一次加载参与最多运算；
  c. 延迟隐藏：warp之间，部分负责数据加载，部分负责数据计算；

### V1：TODO

**文件：** `gemm_v1_tiled_smem.cu` — 待重新实现

### V2：TODO

**文件：** `gemm_v2_thread_tiling.cu` — 待重新实现

### V3：TODO

**文件：** `gemm_v3_fastpath.cu` — 待重新实现

### cuBLAS 参考实现

**文件：** `gemm_cublas_ref.cu`

`cublasSgemm` —— NVIDIA 生产级 GEMM 实现，是本项目的性能目标。

## 性能结果

待 V1~V3 重新实现后补充。

## 横向对比

**文件：** `benchmark_all.cu`

当前仅包含 V0 Naive + cuBLAS 对比，V1~V3 重新实现后将补充到此文件。

## NVIDIA 参考 API

- **cuBLAS：** `cublasSgemm` / `cublasGemmEx`
- **cuBLASLt：** `cublasLtMatmul`（更灵活，支持自动调优）
- **CUTLASS：** 基于模板的高性能 GEMM 库

## 产物路径

- 可执行文件：`build/bin/`（含 `gemm_v0~v3`、`gemm_cublas_ref`、`gemm_benchmark_all`）
- 性能 CSV：`data/results/`
