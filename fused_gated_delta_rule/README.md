# Fused Gated Delta Rule CUDA 优化复盘

本文档描述 `fused_gated_delta_rule/` 下各版本内核的设计与结论。

---

## 1. 项目目标

实现一个融合算子，将 Gated Delta Rule 的三个核心步骤合并为单个 CUDA kernel，减少内存搬运：

- **Compute_Decay_Gate** — 计算衰减门控（decay gate），控制历史状态的遗忘速度
- **Compute_Delta_Gate** — 计算增量门控（delta gate），控制新输入的更新强度
- **Recurrent_Delta_Rule_Update** — 递归增量规则更新，包含状态更新和输出生成

---

## 2. 数学表达式

### 2.1 符号定义

| 符号 | 含义 | 维度 |
|------|------|------|
| $x$ | 输入序列 | $(B, L, D)$ |
| $W_{decay}$ | 衰减门控投影权重 | $(D, H)$ |
| $b_{decay}$ | 衰减门控偏置 | $(H)$ |
| $W_{delta}$ | 增量门控投影权重 | $(D, H)$ |
| $b_{delta}$ | 增量门控偏置 | $(H)$ |
| $W_{state}$ | 状态投影权重 | $(D, H)$ |
| $b_{state}$ | 状态偏置 | $(H)$ |
| $\alpha$ | 衰减门控（decay gate） | $(B, L, H)$ |
| $\delta$ | 增量门控（delta gate） | $(B, L, H)$ |
| $s$ | 状态向量（state） | $(B, H)$ |
| $o$ | 输出序列 | $(B, L, H)$ |
| $\sigma$ | Sigmoid 激活函数 | — |
| $\epsilon$ | 数值稳定常数 | $10^{-6}$ |

### 2.2 融合前向计算流程

**Step 1: 计算 Decay Gate**

```
for each (b, t):
    alpha_raw[b, t] = x[b, t] @ W_decay^T + b_decay      # (H,)
    alpha[b, t] = sigmoid(alpha_raw[b, t])                  # (H,), 范围 (0, 1)
```

**Step 2: 计算 Delta Gate**

```
for each (b, t):
    delta_raw[b, t] = x[b, t] @ W_delta^T + b_delta       # (H,)
    delta[b, t] = softplus(delta_raw[b, t])                 # (H,), 范围 (0, +inf)
```

**Step 3: 计算输入到状态的投影**

```
for each (b, t):
    u[b, t] = x[b, t] @ W_state^T + b_state               # (H,)
```

**Step 4: Recurrent Delta Rule Update（状态更新 + 输出）**

```
for each batch b:
    s = 0                                                   # 初始化状态为 0
    for t = 0 to L-1:
        # 状态更新：s_new = alpha * s_old + delta * u
        s = alpha[b, t] * s + delta[b, t] * u[b, t]         # (H,)
        
        # 输出：o = s（或额外的输出投影）
        o[b, t] = s                                         # (H,)
```

### 2.3 完整数学公式

**Decay Gate：**

$$
\alpha_{b,t,h} = \sigma\left(\sum_{d=0}^{D-1} x_{b,t,d} \cdot W_{decay,h,d} + b_{decay,h}\right)
$$

**Delta Gate：**

$$
\delta_{b,t,h} = \text{softplus}\left(\sum_{d=0}^{D-1} x_{b,t,d} \cdot W_{delta,h,d} + b_{delta,h}\right)
$$

**State Projection：**

$$
u_{b,t,h} = \sum_{d=0}^{D-1} x_{b,t,d} \cdot W_{state,h,d} + b_{state,h}
$$

**Recurrent State Update：**

$$
s_{b,t,h} = \alpha_{b,t,h} \cdot s_{b,t-1,h} + \delta_{b,t,h} \cdot u_{b,t,h}
$$

其中 $s_{b,-1,h} = 0$（初始状态为零）。

**Output：**

$$
o_{b,t,h} = s_{b,t,h}
$$

### 2.4 Softplus 激活函数

$$
\text{softplus}(x) = \ln(1 + e^x)
$$

等价于数值稳定形式：

$$
\text{softplus}(x) = \max(0, x) + \ln(1 + e^{-|x|})
$$

### 2.5 融合优势

将三个步骤融合为单个 kernel 的优势：

1. **减少全局内存搬运**：$\alpha, \delta, u$ 可以直接在寄存器/共享内存中计算并使用，无需写回全局内存
2. **隐藏延迟**：门控计算和状态更新可以流水线化
3. **更好的占用率**：单个 kernel 可以更好地利用 GPU 资源

---

## 3. 版本演进

### 3.1 v0：朴素基线（逐个操作分离）

**文件：** `fused_gated_delta_rule_v0.cu`

- 每个操作单独实现为 CUDA kernel，中间结果写回全局内存
- Decay Gate、Delta Gate、State Projection 分别计算
- Recurrent Update 单独一个 kernel
- 用于验证正确性和作为后续融合优化的基准

---

## 4. 产物路径

- **可执行文件：** `build/bin/fused_gated_delta_rule_v0`
- **结果 CSV：** `data/results/fused_gated_delta_rule_v0_results.csv`

---

## PTX / SASS

PTX 和 SASS 在 `fused_gated_delta_rule/asm/` 下。

使用 `cuobjdump -sass` 可查看 SASS 反汇编。

## 产物路径

- 可执行文件：`build/bin/`
- 结果 CSV：`data/results/`
- PTX/SASS：`fused_gated_delta_rule/asm/ptx/`、`fused_gated_delta_rule/asm/sass/`
