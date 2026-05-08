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

### 2.3 融合前向计算流程

**Step 1: 计算 Q 的 L2 范数**

```
for each (b, n_q):
    norm = sqrt(sum_{h=0}^{H_q-1} Q[b, n_q, h]^2)
    for each h:
        Q_hat[b, n_q, h] = Q[b, n_q, h] / (norm + eps)
```

**Step 2: 计算 K 的 L2 范数**

```
for each (b, n_k):
    norm = sqrt(sum_{h=0}^{H_k-1} K[b, n_k, h]^2)
    for each h:
        K_hat[b, n_k, h] = K[b, n_k, h] / (norm + eps)
```

---

## 3. 版本演进

### 3.1 v0：朴素基线（逐个操作分离）

**文件：** `fused_l2_norm_qk_v0.cu`

- Q 和 K 的 L2 归一化分别实现为独立的 CUDA kernel
- 用于验证正确性和作为后续融合优化的基准

---

## 4. 产物路径

- **可执行文件：** `build/bin/fused_l2_norm_qk_v0`
- **结果 CSV：** `data/results/fused_l2_norm_qk_v0_results.csv`

---

## PTX / SASS

PTX 和 SASS 在 `fused_l2_norm_qk/asm/` 下。

## 产物路径

- 可执行文件：`build/bin/`
- 结果 CSV：`data/results/`
- PTX/SASS：`fused_l2_norm_qk/asm/ptx/`、`fused_l2_norm_qk/asm/sass/`
