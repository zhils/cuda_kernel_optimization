# Fused Gated Delta Rule

## 目标

将 Gated Delta Rule 的三个核心步骤合为一个 CUDA kernel，减少中间张量全局内存读写：

1. **Compute Decay Gate** — 计算衰减门控（sigmoid），控制历史状态遗忘速度
2. **Compute Delta Gate** — 计算增量门控（softplus），控制新输入更新强度
3. **Recurrent Delta Update** — 递归状态更新和输出生成

---

## 数学定义

### Decay Gate

$$
\alpha_{b,t,h} = \sigma\left(\sum_{d} x_{b,t,d} \cdot W_{decay,h,d} + b_{decay,h}\right)
$$
纯文本：`alpha[b,t,h] = sigmoid(sum_d(x[b,t,d] * W_decay[h,d]) + b_decay[h])`。

### Delta Gate

$$
\delta_{b,t,h} = \text{softplus}\left(\sum_{d} x_{b,t,d} \cdot W_{delta,h,d} + b_{delta,h}\right)
$$
纯文本：`delta[b,t,h] = softplus(sum_d(x[b,t,d] * W_delta[h,d]) + b_delta[h])`。

### State Projection

$$
u_{b,t,h} = \sum_{d} x_{b,t,d} \cdot W_{state,h,d} + b_{state,h}
$$
纯文本：`u[b,t,h] = sum_d(x[b,t,d] * W_state[h,d]) + b_state[h]`。

### Recurrent State Update

$$
s_{b,t,h} = \alpha_{b,t,h} \cdot s_{b,t-1,h} + \delta_{b,t,h} \cdot u_{b,t,h}
$$
纯文本：`s[b,t,h] = alpha[b,t,h] * s[b,t-1,h] + delta[b,t,h] * u[b,t,h]`。

---

## 版本演进

### v0：基线（逐个操作分离）

每个步骤独立为 CUDA kernel，中间结果写回全局内存。

```
5 kernels: W_decay @ x → alpha (sigmoid)
           W_delta @ x → delta (softplus)
           W_state @ x → u
           alpha * s + delta * u → s (recurrent)
           s → output
```

### v1：访存效率优化（单 kernel 全融合 + SMEM 缓存 + float4）

将 5 个 kernel 融合为 **单个 kernel**，grid(B) 每 block 处理一个 batch：

| 优化 | 说明 |
|------|------|
| **全融合** | 投影 + 递联合并，消除 alpha/delta/u 三个中间缓冲（B×L×H ×3） |
| **SMEM 缓存 x** | 每时间步将 x[t][D] 加载到 SMEM，所有 h 线程共享，加载仅 1 次 |
| **float4 向量化** | float4 向量化加载 x 到 SMEM，指令数降为 1/4 |
| **寄存器状态** | 状态 s 在寄存器中跨时间步递推，不写回全局内存 |
| **1 次 syncthreads** | 每时间步仅需 1 次 barrier（SMEM 加载后 + 计算后各 1 次） |

**预期效果：** kernel launch 从 4 次降至 1 次，中间缓冲全局读写减少 3×B×L×H×4B，访存总量降低约 60%。

### v2：计算强度优化（双 head ILP + FMA + float4 权重加载）

在 v1 基础上，核心优化方向是提升计算吞吐：

| 优化 | 说明 |
|------|------|
| **双 head ILP** | 每线程处理 2 个 h，6 路（3 投影 ×2h）累加器交错调度 |
| **`__fmaf_rn`** | 融合乘加 `A += x * W` → `__fmaf_rn(x, W, A)`，指令数减半 |
| **float4 权重加载** | `float4` 一次加载 4 个 weight，与 x 的 4 个元素逐次 FMA |
| **`#pragma unroll 4`** | 展开内层 matvec 循环，暴露 ILP |
| **block=128** | 减少线程数，增加每线程计算密度（每线程 h 数翻倍） |

| 版本 | 策略 | Kernel 数 | 中间缓冲 | SMEM 用途 | 计算方式 | 每线程 h |
|:----|------|:---------:|:--------:|:---------:|:--------:|:--------:|
| v0 | 分离 kernel | 4 | alpha, delta, u | 仅归约 | 标量乘加 | 1 |
| v1 | 全融合 | 1 | 无 | s_x[D] 缓存 x | 标量乘加 | 1 |
| v2 | ILP + FMA | 1 | 无 | s_x[D] 缓存 x | **float4 FMA** | **2** |

---

## INT8 量化补偿实验

对三张投影权重做 INT8 量化的 CPU 仿真精度实验：

| 方案 | 说明 |
|------|------|
| A | per-tensor INT8 反量化后计算 |
| B | per-channel（按输出头）INT8 |
| C | A + per-head output bias correction |
| D | A + state 投影分支保留 FP32 残差 |

构建运行：

```bash
cmake --build build --target fused_gated_delta_rule_compensation_test
./build/bin/fused_gated_delta_rule_compensation_test
```

---

## Nsight Compute 瓶颈分析

`ncu --set basic`，`fused_gated_delta_rule_v0`：

| 内核 | Duration(us) | Compute(SM) | DRAM | Memory | Achieved Occupancy | Reg/Thr |
|:----|:-----------:|:-----------:|:----:|:------:|:------------------:|:-------:|
| `RecurrentDeltaRuleKernel` (v0) | 814.21 | 2.28% | 31.69% | 31.69% | 16.53% | 40 |
| `fused_gdr_v1_kernel` (v1) | 368.17¹ | 1.70% | 0.53% | 22.08% | 16.67% | 40 |
| `fused_gdr_v2_kernel` (v2) | 145.94¹ | 0.09% | 0.61% | 29.22% | 8.33% | 78 |

¹ v1 数值为 4×1024×512×256 配置，v2 为 8×2048×512×256 配置。

时间维串行递推导致并行度受限，为延迟/带宽混合瓶颈。v1/v2 通过消除中间缓冲与 ILP 优化提升有效吞吐。

---

## Warp Stall 原因分析

| 版本 | #1 Stall | #2 Stall | #3 Stall | #4 Stall | #5 Stall |
|:----|:---------|:---------|:---------|:---------|:---------|
| v0 | **Long Scoreboard 86.3%** | Wait 10.8% | Short Scoreboard 2.0% | Not Selected 0.4% | Math Pipe Throttle 0.3% |
| v1 | **Long Scoreboard 96.7%** | Wait 3.0% | No Instruction 0.1% | Short Scoreboard 0.1% | Math Pipe Throttle 0.1% |
| v2 | **Long Scoreboard 96.6%** | Wait 2.9% | Short Scoreboard 0.4% | No Instruction 0.1% | — |

三个版本均以 Long Scoreboard 占绝对主导（86-97%），反映 Gated Delta Rule 的时间维串行递推特性：每个时间步都要从全局内存重新读取权重矩阵，每次读取都触发大量长延迟全局加载。v1/v2 的融合优化减小了中间缓冲的读写，但对权重矩阵的全局加载没有改善。

## 构建

```bash
cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=120 && make fused_gated_delta_rule_v0 -j$(nproc)
cd ..
./build/bin/fused_gated_delta_rule_v0

# v1/v2 同理
./build/bin/fused_gated_delta_rule_v1
./build/bin/fused_gated_delta_rule_v2
```

---

## 主场景性能口径（统一）

主指标统一为主场景 `gpu_ms`，NCU 吞吐仅用于瓶颈归因。

| 实现 | 主场景维度 | GPU耗时(ms) | 校验状态 | 数据文件 |
|---|---|---:|---|---|
| `fused_gated_delta_rule_v2` | `B=8,L=2048,D=512,H=256` | 109.07 | PASS | `data/results/fused_gated_delta_rule_v2_results.csv` |

环境口径：`RTX 5060 Ti (sm_120) + CUDA 13.2`。
统一汇总：`data/results/main_scenario_unified.csv`（retest tag: `20260512_manual_retest`）。

## 已知边界与后续补充

- 当前重点在前向递推与融合访存优化，未覆盖训练反向路径。
- 建议增加更长序列（`L>=4096`）下的稳定性与吞吐趋势测试。
