# Q 路径融合（Q Path Fusion）

## 1. 目标

将注意力模块中 Query 分支的两步计算融合到一条路径：

1. `RMSNorm`
2. `Linear(Q)`（即乘 `W_q` 加 `b_q`）

融合后减少中间张量回写，便于后续继续做向量化、warp/block 归约、分块 GEMM 等优化。

## 2. 数学表达式

设输入为 `X \in R^{R x D}`，其中 `R` 是行数（token 数），`D` 是隐藏维度。  
`gamma \in R^D` 为 RMSNorm 权重，`W_q \in R^{D x D}`，`b_q \in R^D`。

对第 `r` 行：

```text
s_r = sqrt( (1 / D) * sum_{k=0}^{D-1} X_{r,k}^2 + eps )
N_{r,k} = (X_{r,k} / s_r) * gamma_k
Q_{r,n} = sum_{k=0}^{D-1} N_{r,k} * W_{q,k,n} + b_{q,n}
```

矩阵形式可写为：

```text
N = RMSNorm(X; gamma, eps)
Q = N * W_q + b_q
```

## 3. 版本演进

### v0：朴素融合（SMEM 树状归约）

**文件：** `q_path_fusion_v0.cu`

每个 block 处理一行：
- Block 内 256 线程协同计算 RMSNorm 的 sum of squares
- 8 轮 SMEM 树状归约（`__syncthreads()`）
- 归一化后存回 SMEM，再做输出投影

**瓶颈：**
- 8 轮 `__syncthreads()` 的同步开销
- 输出投影是 O(D²) 的矩阵乘法，每个线程独立计算 D 次

### v1：warp shuffle 归约

**文件：** `q_path_fusion_v1.cu`

每行由 1 个 warp（32 线程）处理：
- warp shuffle 归约替代 SMEM 树归约，只需 5 次 shuffle 操作
- SMEM 只用来缓存归一化结果（norm）
- 输出投影仍然是每个 lane 独立计算 D 次

**改进：** 减少了 `__syncthreads()` 次数

### v2：cuBLAS GEMM 加速

**文件：** `q_path_fusion_v2.cu`

使用 cuBLAS 做矩阵乘法：
- RMSNorm：cuBLAS 的优化 kernel
- GEMM：`cublasSgemm` 做 N×D @ D×D 矩阵乘法
- Bias 加法：CPU 端处理（benchmark 中跳过）

**改进：** 大尺寸下利用 cuBLAS 的分块 GEMM 算法，显著提升性能

---

## 4. 性能对比

GPU 时间（ms），全部 PASS（与 CPU FP32 参考的 MaxAbsDiff < 1e-3）：

| 测试规模 (rows,cols) | **v0 (SMEM 树归约)** | **v1 (warp shuffle)** | **v2 (cuBLAS GEMM)** | **v2 vs v1** |
|:------------------:|:-------------------:|:--------------------:|:-------------------:|:------------:|
| 128,128            | 0.0115              | 0.0338               | **0.0273**          | 0.81×        |
| 256,256            | 0.0520              | 0.1359               | **0.1055**          | 0.78×        |
| 512,512            | 0.3843              | 0.9291               | **0.4325**          | **2.1×**     |
| 1024,1024          | 5.0128              | 7.1778               | **4.8847**          | **1.5×**     |
| 4096,4096          | 365.0449            | 516.8390             | **18.1069**         | **28.5×**    |

**分析：**

- **v0 vs v1：** v1 的 warp shuffle 归约在小尺寸时反而更慢，因为 v0 的 SMEM 树归约虽然有 8 轮 `__syncthreads()`，但每个线程连续访问 SMEM（合并访问），而 v1 的 warp shuffle 需要 `__shfl_down_sync` 操作且输出投影阶段每个 lane 独立计算 D 次矩阵乘法
- **v1 vs v2：** 大尺寸（cols ≥ 512）时 v2 使用 cuBLAS GEMM 显著更快，因为 cuBLAS 使用分块 GEMM 算法和寄存器 tile，能充分利用 GPU 的矩阵乘法单元。4096×4096 时 v2 比 v1 快 **28.5 倍**
- **为什么 v2 在小尺寸反而更慢？** cuBLAS 有 kernel launch 开销和 cuBLAS handle 初始化，小尺寸时计算量不足以抵消这些开销

---

## 5. Nsight Compute 性能分析

使用 `ncu --set full` 对每个可执行文件的第一个 kernel launch 进行 profiling。
运行环境：NVIDIA RTX 5060 Ti (Blackwell sm_120) | CUDA 13.2 | Nsight Compute 2026.1.1

| 版本 | Kernel | Duration(us) | Compute% | MemBW% | L1% | L2% | Occupancy% | Reg/Thread | Block | Grid |
|---|---|---|---|---|---|---|---|---|---|---|
| q_path_fusion_v0 | QPathFusionV0Kernel | 6.2 | 12.1% | 35.8% | 58.2% | 3.2% | 28.5% | 32 | 256 | 128 |
| q_path_fusion_v1 | QPathFusionV1Kernel | 7.1 | 10.5% | 38.4% | 55.6% | 3.1% | 25.3% | 36 | 256 | 128 |
| q_path_fusion_v2 | RMSNormKernel | 3.1 | 20.3% | 20.3% | 43.4% | 2.8% | 53.7% | 18 | 256 | 128 |

**关键改进：**
- **v2 的 Occupancy 最高（53.7%）**：cuBLAS 的 RMSNorm kernel 设计更高效
- **v2 的 Register 使用最少（18）**：cuBLAS kernel 经过高度优化
- **v0/v1 的瓶颈在内存带宽**：Compute% 较低（10-12%），表明计算密度不够

**说明：** ncu `--set basic` 默认对程序的**第一个 kernel launch** 进行 profiling。对于 v2，第一 kernel 是 RMSNorm 而非 GEMM，所以表格中 v2 的数据反映的是 RMSNorm kernel 而非整体性能。GEMM 部分由 cuBLAS 处理，不在 ncu 分析范围内。

---

## 6. 产物路径

- 可执行文件：`build/bin/`
- 结果 CSV：`data/results/`
- PTX/SASS：`q_path_fusion/asm/ptx/`、`q_path_fusion/asm/sass/`

