# Fused Output Norm Gate CUDA 优化复盘

本文档描述 `fused_output_norm_gate/` 下各版本内核的设计与结论。

---

## 1. 项目目标

实现一个融合算子，将输出阶段的门控、归一化、乘法和投影合并为单个 CUDA kernel，减少内存搬运：

- **Gate_Projection + SiLU** — 门控投影后接 SiLU 激活
- **RMSNorm** — 均方根归一化
- **Multiply_Gate** — 门控乘法（逐元素乘）
- **Linear(out_proj)** — 输出线性投影

---

## 2. 数学表达式

### 2.1 符号定义

| 符号 | 含义 | 维度 |
|------|------|------|
| $x$ | 输入序列（来自前层输出） | $(B, L, D_{in})$ |
| $W_{gate}$ | 门控投影权重 | $(D_{in}, H)$ |
| $b_{gate}$ | 门控投影偏置 | $(H)$ |
| $W_{out}$ | 输出投影权重 | $(H, D_{out})$ |
| $b_{out}$ | 输出投影偏置 | $(D_{out})$ |
| $g$ | RMSNorm 缩放参数 | $(H)$ |
| $\epsilon$ | 数值稳定常数 | $10^{-6}$ |
| $gate$ | 门控值（SiLU 激活后） | $(B, L, H)$ |
| $\hat{x}$ | RMSNorm 归一化后 | $(B, L, H)$ |
| $y$ | 门控乘法后 | $(B, L, H)$ |
| $o$ | 最终输出 | $(B, L, D_{out})$ |

### 2.2 融合前向计算流程

**Step 1: Gate Projection + SiLU**

```
for each (b, t):
    gate_raw[b, t] = x[b, t] @ W_gate^T + b_gate      # (H,)
    gate[b, t] = SiLU(gate_raw[b, t])                  # (H,)
```

**Step 2: RMSNorm**

```
for each (b, t):
    # 计算均方根
    rms = sqrt(mean(gate[b, t]^2) + eps)
    # 归一化并缩放
    x_hat[b, t] = gate[b, t] / rms * g                 # (H,)
```

**Step 3: Multiply Gate**

```
for each (b, t):
    y[b, t] = x_hat[b, t] * gate[b, t]                 # (H,), 逐元素乘
```

**Step 4: Linear Output Projection**

```
for each (b, t):
    o[b, t] = y[b, t] @ W_out^T + b_out                # (D_out,)
```

### 2.3 完整数学公式

**Gate Projection + SiLU：**

$$
gate_{b,t,h} = \text{SiLU}\left(\sum_{d=0}^{D_{in}-1} x_{b,t,d} \cdot W_{gate,h,d} + b_{gate,h}\right)
$$

**RMSNorm：**

$$
\text{RMS}(gate_{b,t}) = \sqrt{\frac{1}{H} \sum_{h=0}^{H-1} gate_{b,t,h}^2 + \epsilon}
$$

$$
\hat{x}_{b,t,h} = \frac{gate_{b,t,h}}{\text{RMS}(gate_{b,t})} \cdot g_h
$$

**Multiply Gate：**

$$
y_{b,t,h} = \hat{x}_{b,t,h} \cdot gate_{b,t,h}
$$

**Linear Output Projection：**

$$
o_{b,t,d} = \sum_{h=0}^{H-1} y_{b,t,h} \cdot W_{out,d,h} + b_{out,d}
$$

### 2.4 SiLU 激活函数

$$
\text{SiLU}(x) = x \cdot \sigma(x) = x \cdot \frac{1}{1 + e^{-x}}
$$

### 2.5 融合优势

将四个步骤融合为单个 kernel 的优势：

1. **减少全局内存搬运**：$gate, \hat{x}, y$ 可以直接在寄存器中计算并使用，无需写回全局内存
2. **隐藏延迟**：SiLU、RMSNorm、乘法和投影可以流水线化
3. **更好的占用率**：单个 kernel 可以更好地利用 GPU 资源
4. **减少 kernel launch 开销**：4 个操作合并为 1 个 kernel

---

## 3. 版本演进

### 3.1 v0：朴素基线（逐个操作分离）

**文件：** `fused_output_norm_gate_v0.cu`

- 每个操作单独实现为 CUDA kernel，中间结果写回全局内存
- Gate Projection + SiLU、RMSNorm、Multiply Gate、Linear Output 分别计算
- 用于验证正确性和作为后续融合优化的基准

---

## 4. 产物路径

- **可执行文件：** `build/bin/fused_output_norm_gate_v0`
- **结果 CSV：** `data/results/fused_output_norm_gate_v0_results.csv`

---

## PTX / SASS

PTX 和 SASS 在 `fused_output_norm_gate/asm/` 下。

## 产物路径

- 可执行文件：`build/bin/`
- 结果 CSV：`data/results/`
- PTX/SASS：`fused_output_norm_gate/asm/ptx/`、`fused_output_norm_gate/asm/sass/`
