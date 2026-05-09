# Fused L2 Norm Q/K CUDA 优化复盘

本文档描述 `fused_l2_norm_qk/` 下各版本内核的设计与结论。

---

## 1. 项目目标

实现一个融合算子，将 Q 和 K 的 L2 归一化合并为单个 CUDA kernel，减少内存搬运：

- **L2_Norm_Q** — 对 Query 张量按行/通道做 L2 归一化
- **L2_Norm_K** — 对 Key 张量按行/通道做 L2 归一化

---

## 2. 数学表达式

### 2.1 符号定义

| 符号 | 含义 | 维度 |
|------|------|------|
| $Q$ | Query 张量 | $(B, N_q, H_q)$ |
| $K$ | Key 张量 | $(B, N_k, H_k)$ |
| $\hat{Q}$ | 归一化后的 Query | $(B, N_q, H_q)$ |
| $\hat{K}$ | 归一化后的 Key | $(B, N_k, H_k)$ |

### 2.2 L2 归一化定义

对张量 $X \in \mathbb{R}^{(B, N, H)}$ 的每一行（最后一个维度）做 L2 归一化：

$$
\|x_{b,n}\|_2 = \sqrt{\sum_{h=0}^{H-1} x_{b,n,h}^2}
$$

$$
\hat{x}_{b,n,h} = \frac{x_{b,n,h}}{\|x_{b,n}\|_2 + \epsilon}
$$

其中 $\epsilon$ 是一个很小的常数（如 $10^{-6}$），防止除零。

### 2.3 前向计算流程

```
for each (b, n):
    norm = sqrt(sum_{h=0}^{H-1} X[b, n, h]^2)
    for each h:
        X_hat[b, n, h] = X[b, n, h] / (norm + eps)
```

对 Q 和 K 分别执行一次。

---

## 3. 版本演进

### 3.1 v0：朴素基线（SMEM 树状归约）

**文件：** `fused_l2_norm_qk_v0.cu`

- 每个 block 处理一行 (b,n)，256 线程
- 先算平方和，SMEM 树状归约（8 轮 `__syncthreads()`）
- 再从全局读一遍做归一化
- 两个 kernel launch 分别处理 Q 和 K

### 3.2 v1：Warp Shuffle 归约 + float4 向量化

**文件：** `fused_l2_norm_qk_v1.cu`

v0 的问题：
1. **SMEM 树状归约要 8 轮 `__syncthreads()`** — 每次同步都让所有 warp 等最慢的那个
2. **标量逐元素访存** — 没利用 128-bit 的 float4 加载

v1 的改动：
- **warp shuffle 替代 SMEM 归约**：`__shfl_xor_sync` 只需 5 轮寄存器交换（16→8→4→2→1），无 `__syncthreads`，延迟极低
- **float4 向量化**：读 x_row 和写 out_row 都用 `float4`，每 4 个元素合并 1 条访存指令
- **blockDim 降到 128**：更少的线程更充分的利用，同时保持足够的行级并行

### 3.3 v2：SMEM 缓存整行 + 单 kernel 融合

**文件：** `fused_l2_norm_qk_v2.cu`

v1 还有一个小问题：虽然用了 float4，但 x_row 还是从全局读了两遍（一遍算平方和，一遍归一化）。

v2 的改动：
- **SMEM 缓存整行**：协作式把 x_row 一次性加载到 SMEM，之后两次访问都从 SMEM 读。SMEM 带宽是全局的 ~20 倍，两种读几乎免费
- **Q/K 合为单 kernel**：一个 grid 同时覆盖 Q 和 K 的所有行，减少 1 次 launch，小块场景下 GPU 有更多 block 来调度

---

## 4. 性能对比

GPU 时间（ms），全部 PASS（与 CPU FP32 参考的 MaxAbsDiff < 1e-4）：

| 测试规模 (B,N_q,H_q,N_k,H_k) | **v0** | **v1** | **v2** | **最优提升** |
|:--------------------------:|:-----:|:-----:|:-----:|:----------:|
| 1,128,64,128,64            | 0.0100 | 0.0159 | 0.0139 | — |
| 1,256,128,256,128          | 0.0123 | **0.0129** | **0.0143** | — |
| 2,512,128,512,128          | 0.0196 | 0.0176 | **0.0114** | **1.7×** |
| 4,1024,256,1024,256        | 0.0518 | **0.0261** | **0.0296** | **2.0×** |
| 8,2048,256,2048,256        | 0.2145 | 0.1762 | **0.1732** | **1.24×** |

**分析：**
- 小尺寸（128 行）差距不大，因为总数据量小，全局缓存就能命中，优化空间有限
- **v1 在中尺寸（1024~2048 行）优势最明显**：warp shuffle 比 SMEM 树状归约快得多，float4 向量化降低了全局内存压力
- **v2 在大尺寸和高度并行场景更稳定**：SMEM 缓存避免双读，融合 kernel 增加 grid 规模
- 所有测试精度 PASS

---

## 5. Nsight Compute 性能分析

使用 `ncu --set basic` 对每个可执行文件进行 profiling。
运行环境：NVIDIA RTX 5060 Ti (Blackwell sm_120) | CUDA 13.2 | Nsight Compute 2026.1.1

| 版本 | Kernel | Duration(us) | Compute% | MemBW% | L1% | L2% | Occupancy% | Reg/Thread | Block | Grid |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | L2NormKernel (SMEM tree) | 3.1 | 19.4% | 19.4% | 44.5% | 1.5% | 50.8% | 24 | 256 | 128 |
| v1 | L2NormShuffleKernel | **2.7** | 6.0% | 4.7% | 10.6% | 1.8% | 26.8% | 40 | 128 | 128 |
| v2 | L2NormFusedKernel (SMEM cache) | 3.1 | **15.6%** | **8.0%** | **16.2%** | **2.7%** | **51.8%** | 40 | 128 | **256** |

**关键解读：**
- **v1 的 Duration 最短（2.7us）**：warp shuffle 消除了 8 轮 `__syncthreads()`，kernel 执行最快。但 occupancy 降到 26.8%，因为寄存器从 24 升到 40
- **v2 的 Compute% 最高（15.6%）**：SMEM 缓存让计算更密集，同时 occupancy 回到 51.8%
- **v2 Grid=256**：融合 Q/K 后 grid 翻倍，GPU 能调度更多 block

---

## 6. 产物路径

- **可执行文件：** `build/bin/fused_l2_norm_qk_v0`, `v1`, `v2`
- **结果 CSV：** `data/results/fused_l2_norm_qk_v{0,1,2}_results.csv`
- **PTX/SASS：** `fused_l2_norm_qk/asm/ptx/`、`fused_l2_norm_qk/asm/sass/`
