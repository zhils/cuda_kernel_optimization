# Fused Output Norm Gate

## 目标

将输出阶段的门控、归一化、门控乘法和输出投影四个步骤融合为单个 CUDA kernel，减少中间张量全局内存读写：

1. **Gate Projection + SiLU** — 门控投影 + SiLU 激活
2. **RMSNorm** — 均方根归一化
3. **Multiply Gate** — 逐元素门控乘法
4. **Linear Output Projection** — 输出线性投影

---

## 数学定义

### Gate Projection + SiLU

$$
gate_{b,t,h} = \text{SiLU}\left(\sum_{d} x_{b,t,d} \cdot W_{gate,h,d} + b_{gate,h}\right)
$$

### RMSNorm

$$
\text{RMS}(gate_{b,t}) = \sqrt{\frac{1}{H} \sum_h gate_{b,t,h}^2 + \epsilon}
$$

$$
\hat{x}_{b,t,h} = \frac{gate_{b,t,h}}{\text{RMS}(gate_{b,t})} \cdot g_h
$$

### Multiply Gate

$$
y_{b,t,h} = \hat{x}_{b,t,h} \cdot gate_{b,t,h}
$$

### Linear Output Projection

$$
o_{b,t,d} = \sum_h y_{b,t,h} \cdot W_{out,d,h} + b_{out,d}
$$

---

## 版本演进

### v0：基线（4 个分离 kernel）

每个操作独立实现为 CUDA kernel，中间结果写回全局内存。

### v1：访存效率优化（单 kernel 全融合）

将 4 个操作融合为 **单个 kernel**，每 block 处理一行 (b,t)：

- 消除 3 个中间全局缓冲区（gate / x_hat / y）
- float4 向量化加载 x 到 SMEM，降低指令数
- SMEM 双缓冲：s_x[D_in] 缓存输入，s_gate[H] 缓存门控 / y 值
- Warp Shuffle 完成 RMSNorm 归约，避免 SMEM 树归约延迟
- 输出投影从 SMEM 读 y，减少全局内存往返

**预期效果：** 消除中间张量（gate: B×L×H, x_hat: B×L×H, y: B×L×H）的全局内存读写，访存总量降低约 3×。

### v2：计算强度优化（2 行/block + 权重复用）

在 v1 基础上，每 block 处理 **2 行**，核心思路是 **权重复用**：

- `W_gate` 和 `W_out` 对所有 (b,t) 位置共享，处理 2 行时读取一次但计算两次
- float4 向量化加载 `W_gate` / `W_out`，配合 `__fmaf_rn` 融合乘加
- `#pragma unroll` 展开内层 matvec 循环，增加指令级并行
- 分离加载/计算阶段，消除 bank conflict

**预期效果：** 权重矩阵的算术强度翻倍（FLOPs/Byte ×2），更适合计算受限场景。

| 版本 | 策略 | 中间缓冲 | 每 block 行数 | 权重复用 | 向量化 |
|:----|------|:--------:|:------------:|:--------:|:------:|
| v0 | 4 个分离 kernel | gate, x_hat, y | 1 | 无 | 无 |
| v1 | 单 kernel 全融合 | 无 | 1 | 隐式（SMEM） | float4(x, output) |
| v2 | 2 行/block + 权重复用 | 无 | 2 | 显式（寄存器） | float4(x, W, output) + FMA |

---

## Nsight Compute 瓶颈分析

`ncu --set basic`，`fused_output_norm_gate_v0`：

| 内核 | Duration(us) | Compute(SM) | DRAM | Memory | Achieved Occupancy | Reg/Thr |
|:----|:-----------:|:-----------:|:----:|:------:|:------------------:|:-------:|
| `LinearOutputKernel` | 385.38 | 11.73% | 0.49% | 95.81% | 90.97% | 38 |

Occupancy 高、缓存流量高，主受访存路径与数据重用模式影响。v1/v2 将消除中间张量全局读写，提升有效访存带宽利用率。

---

## 构建

```bash
cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=120 && make fused_output_norm_gate_v0 -j$(nproc)
cd ..
./build/bin/fused_output_norm_gate_v0

# v1/v2 同理
./build/bin/fused_output_norm_gate_v1
./build/bin/fused_output_norm_gate_v2
```
