# Fused Conv1D + SiLU CUDA 优化复盘

本文档描述 `fused_conv1d_silu/` 下各版本内核的设计与结论。

---

## 1. 项目目标

实现一个融合算子，将以下多个操作合并为单个 CUDA kernel，减少内存搬运：

- **Linear(in_proj_qkv)** — 输入投影到 QKV 空间
- **Linear(in_proj_z)**, **Linear(in_proj_a)**, **Linear(in_proj_b)** — 门控投影
- **Causal_Conv1d** — 因果一维卷积（只依赖历史信息）
- **SiLU_Activation** — SiLU 激活函数
- **Split_QKV** — 将结果切分为 Q、K、V

---

## 2. 数学表达式

### 2.1 符号定义

| 符号 | 含义 | 维度 |
|------|------|------|
| $x$ | 输入序列 | $(B, L, D)$ |
| $W_{qkv}$ | QKV 投影权重 | $(D, 3 \cdot H)$ |
| $b_{qkv}$ | QKV 投影偏置 | $(3 \cdot H)$ |
| $W_z, W_a, W_b$ | 门控投影权重 | $(D, H)$ |
| $b_z, b_a, b_b$ | 门控投影偏置 | $(H)$ |
| $K_{conv}$ | 因果卷积核 | $(k_{size}, H)$ |
| $Q, K, V$ | 输出 QKV | $(B, L, H)$ |

### 2.2 前向计算流程

**Step 1: 线性投影**

```
qkv_proj = x @ W_qkv^T + b_qkv          # (B, L, 3*H)

z_proj = x @ W_z^T + b_z                # (B, L, H)
a_proj = x @ W_a^T + b_a                # (B, L, H)
b_proj = x @ W_b^T + b_b                # (B, L, H)
```

**Step 2: Split QKV**

```
Q_raw, K_raw, V_raw = split(qkv_proj, 3, dim=-1)   # each (B, L, H)
```

**Step 3: 门控分支的因果卷积 + SiLU**

```
# Causal Conv1D: 每个位置 t 只依赖 [t-k_size+1, t]
# 对 z_proj 做因果卷积
z_conv[t, h] = sum_{i=0}^{k_size-1} K_{conv}[i, h] * z_proj[t - i, h]
               (其中 t-i < 0 时取 0，即 padding=0)

# SiLU 激活
z_act = SiLU(z_conv) = z_conv * sigmoid(z_conv)

# 对 a_proj, b_proj 同样处理
a_conv = CausalConv1d(a_proj, K_conv)
a_act = SiLU(a_conv)

b_conv = CausalConv1d(b_proj, K_conv)
b_act = SiLU(b_conv)
```

**Step 4: 融合输出**

```
# 典型的 Mamba/SSM 风格融合
V = V_raw * z_act                        # 门控调制 V
Q = Q_raw                                # Q 保持不变
K = K_raw                                # K 保持不变

# 或更复杂的融合（如 a_act, b_act 参与状态更新）
# 这里以标准融合为例
```

### 2.3 SiLU 激活函数

$$
\text{SiLU}(x) = x \cdot \sigma(x) = x \cdot \frac{1}{1 + e^{-x}}
$$

### 2.4 Causal Conv1D 的数学定义

对于输入 $u \in \mathbb{R}^{(B, L, H)}$，卷积核 $K \in \mathbb{R}^{(k_{size}, H)}$：

$$
y_{b,t,h} = \sum_{i=0}^{\min(t, k_{size}-1)} K_{i,h} \cdot u_{b, t-i, h}
$$

其中 $b$ 是 batch，$t$ 是时间步，$h$ 是通道。

---

## 3. 版本演进

### 3.1 v0：朴素基线（逐个操作分离）

**文件：** `fused_conv1d_silu_v0.cu`

- 每个操作单独实现为 CUDA kernel，中间结果写回全局内存
- 用于验证正确性和作为后续融合优化的基准

---

## 4. 产物路径

- **可执行文件：** `build/bin/fused_conv1d_silu_v0`
- **结果 CSV：** `data/results/fused_conv1d_silu_v0_results.csv`
