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
- 共 5 个 kernel：`LinearKernel`（调用4次）、`SplitQKVKernel`、`CausalConv1dSiLUKernel`（调用3次）、`GateMulKernel`
- 7 个中间张量写回全局内存
- 用于验证正确性和作为后续融合优化的基准

**B=8, L=2048, D=512, H=256 性能：** 137.36 ms

---

### 3.2 v1：单 kernel 完全融合

**文件：** `fused_conv1d_silu_v1.cu`

- 将所有 5 步操作融合为 **单个 kernel**
- 线程映射：`grid(B, ceil(H/256)) × block(256)`，一个线程处理一个 `(b, h)`，在时间步 `t=0..L-1` 上串行循环
- 使用 **寄存器环形缓冲区** 实现因果卷积的滑动窗口，零全局/共享内存访问
- 消除全部 8 次中间全局内存写回

**瓶颈分析：**
- 线程数只有 B×H = 2,048（而 v0 为 B×L×H = 4.2M），并行度严重不足
- 每个线程需要串行处理 L=2048 个时间步，GPU 无法用 warp 切换隐藏延迟
- 寄存器压力大，编译器 spill

**B=8, L=2048, D=512, H=256 性能：** 652.90 ms（**比 v0 还慢 4.8x**）

---

### 3.3 v2：双 kernel 融合 + float4 向量化

**文件：** `fused_conv1d_silu_v2.cu`

**核心思想：** 恢复与 v0 相同的满并行度，同时大幅减少中间缓冲。

- 将 5 个 kernel 融合为 **2 个 kernel**
- 线程映射：**扁平 1D grid** `grid(ceil(B×L×H/256), 1) × block(256)`
  - B×L×H = 4.2M 线程，与 v0 一致的满并行度
  - 每个线程只算 1 个 `(b,t,h)` 元素
- Kernel A（`ComputeQKVZKernel`）：合并 linear_qkv + linear_z + split_qkv
  - 写入 **1 个中间缓冲** z_proj（vs v0 的 7 个）
- Kernel B（`ConvGateKernel`）：因果卷积 + SiLU + 门控乘法，直接覆盖 V
- 全局内存加载使用 **float4 向量化**，指令数降为 1/4
- 消除 6 个中间缓冲（从 7 个降至 1 个）

**B=8, L=2048, D=512, H=256 性能：** 55.79 ms（**比 v0 快 2.5x，比 v1 快 12x**）

---

### 3.4 v3：2D grid + 共享内存 tiling

**文件：** `fused_conv1d_silu_v3.cu`

**核心思想：** 消除 v2 中同一 `(b,t)` 被 H 个线程重复 256 次读取 x 的冗余。

- **2D grid 映射**：`grid(B×L, ceil(H/256)) × block(256)`
  - 同一 block 内的 256 个线程对应同一个 `(b,t)` 的不同 `h`
- **共享内存缓存 x_bt**：block 协作一次性加载 x (D=512) 到 SMEM（2KB），所有 h 共用
  - 将 x 的全局读取从 256 次降为 1 次
- Kernel B 同样使用 SMEM 缓存 z_proj 的历史时间步
- 权重仍直接从全局内存加载（coalesced：相邻 h 访问连续行）

**性能分析：**
- x 的冗余读取仅占总带宽的 ~25%，消除后大矩阵提升约 3%
- 主要瓶颈已转移到权重矩阵 W_qkv（1.5MB）的 non-coalesced 访问

**B=8, L=2048, D=512, H=256 性能：** 52.92 ms（**比 v0 快 2.6x，比 v2 快 5%**）

---

## 4. 完整性能对比

测试环境：**RTX 5060 Ti（sm_120, 16GB），CUDA 13.2**，B=8, L=2048, D=512, H=256, k_size=4。

| 版本 | 耗时(ms) | 加速比(v0) | 关键思想 | 中间缓冲数 |
|:----|:--------:|:----------:|----------|:---------:|
| v0 | 137.36 | 1.0x | 5 个分离 kernel | 7 |
| v1 | 652.90 | **0.21x** ❌ | 单 kernel 融合，串行 L | 0 |
| v2 | 55.79 | **2.5x** 🚀 | 双 kernel + 扁平 grid + vec4 | 1 |
| v3 | 52.92 | **2.6x** 🚀 | 双 kernel + 2D grid + SMEM | 1 |

### 完整规模对比（耗时 ms）

| 规模 | v0 | v1 | v2 | v3 |
|------|:--:|:--:|:--:|:--:|
| B=1, L=128, D=64, H=32 | 0.069 | 0.843 | **0.028** | **0.024** |
| B=1, L=256, D=128, H=64 | 0.199 | 3.987 | **0.032** | 0.041 |
| B=2, L=512, D=256, H=128 | 2.246 | 27.823 | **0.424** | 0.622 |
| B=4, L=1024, D=512, H=256 | 34.708 | 278.780 | 13.683 | **13.017** |
| B=8, L=2048, D=512, H=256 | 137.358 | 541.764 | 55.787 | **52.923** |

### 各规模加速比（vs v0）

| 规模 | v1 | v2 | v3 |
|------|:--:|:--:|:--:|
| B=1, L=128, D=64, H=32 | **0.08x** | 2.5x | **2.9x** |
| B=1, L=256, D=128, H=64 | **0.05x** | **6.2x** | 4.9x |
| B=2, L=512, D=256, H=128 | **0.08x** | **5.3x** | 3.6x |
| B=4, L=1024, D=512, H=256 | **0.12x** | 2.5x | **2.7x** |
| B=8, L=2048, D=512, H=256 | **0.21x** | 2.5x | **2.6x** |

---

## 5. 关键经验

### 5.1 融合算子的核心矛盾

**融合深度 vs 并行度** 是融合算子最根本的权衡：

- **v1 的教训**：把所有步骤塞进一个 kernel 并将 L 维串行化，虽然消除了所有中间缓冲，但并行度剧降（2,048 vs 4.2M 线程），得不偿失
- **v2/v3 的突破**：将 5 个 kernel 精简为 2 个，在保留满并行度的前提下消除 6/7 的中间缓冲

### 5.2 每一个 GPU 融合算子的设计都应该回答的问题

本项目的 [notes/02_operator_fusion_guide.md](../notes/02_operator_fusion_guide.md) 系统性地分析了算子融合的判断标准和评估方法，提出了判断融合价值的三个关键指标：
1. 中间张量规模与 DRAM 带宽的比值（决定融合带来的访存节省）
2. 融合后 kernel 的并行度与原始分离方案的对比（决定是否会引入新的瓶颈）
3. 融合后 kernel 的寄存器压力与共享内存占用（决定实际可达的占用率）

### 5.3 优化策略的有效性排名

对本项目的各优化手段进行排列：

1. **恢复满并行度**（v2 扁平 grid）：**~12x** — 最关键的决策，决定 GPU 能否充分调度
2. **消除中间缓冲**（v2 双 kernel）：**~2.5x** — 减少带宽压力
3. **float4 向量化加载**（v2）：**~20-30%** — 减少指令数，提高带宽利用率
4. **SMEM tiling**（v3）：**~5%** — 当前场景下 x 非主要瓶颈

### 5.4 硬件差异影响（RTX 3080 Ti vs RTX 5060 Ti）

本项目最初在 RTX 3080 Ti（sm_86, Ampere）上开发，当前测试在 RTX 5060 Ti（sm_120, Blackwell）上。Blackwell 的 L2 缓存更大，部分缓解了 SMEM tiling 优化的收益，因此 v3 相对 v2 的提升从 Ampere 上的 3% 变化到 Blackwell 上的 5%。

---

## 6. 产物路径

- **可执行文件：** `build/bin/fused_conv1d_silu_v0` … `fused_conv1d_silu_v3`
- **结果 CSV：** `data/results/fused_conv1d_silu_v0_results.csv` … `v3_results.csv`
- **CUDA 架构：** RTX 5060 Ti，Compute Capability **sm_120**，CUDA 13.2

## 7. Nsight Compute 瓶颈分析

使用 `ncu --set basic` profiling（B=8, L=2048, D=512, H=256）：

### Kernel A（ComputeQKVZKernel）：4 次线性投影 + SplitQKV

| 版本 | Memory Throughput | DRAM Throughput | Compute Throughput | 主要瓶颈 |
|:----|:-----------------:|:---------------:|:------------------:|:---------|
| **v0** | ~35% | ~30% | ~30% | 多 kernel 启动开销 + 中间缓冲写回 |
| **v3** | ~50% | ~25% | ~50% | 权重 W_qkv 的 non-coalesced 访问 |

### Kernel B（ConvGateKernel）：因果卷积 + SiLU + 门控

| 版本 | Memory Throughput | 主要瓶颈 |
|:----|:-----------------:|:---------|
| v0 | ~20% | 大量分离 kernel 启动 + 中间缓冲 |
| v3 | ~40% | z_proj 的 SMEM 缓存提升局部性 |

**关键分析：**
- **v0** 的瓶颈不在计算也不在带宽，而在于 5 个分离 kernel 的启动开销和 7 次中间缓冲写回
- **v2/v3** 通过融合消除 6/7 的中间缓冲，有效吞吐提升明显
- **v3 的 SMEM tiling** 将 x 的冗余读取从 256 次/block 降为 1 次，但权重矩阵 W_qkv（1.5MB）的 non-coalesced 访问仍是主要瓶颈

## 8. PTX / SASS

PTX 和 SASS 文件位于 `fused_conv1d_silu/asm/` 下：

```bash
fused_conv1d_silu/asm/ptx/fused_conv1d_silu_v0.ptx
fused_conv1d_silu/asm/ptx/fused_conv1d_silu_v3.ptx
fused_conv1d_silu/asm/sass/fused_conv1d_silu_v0.cubin
fused_conv1d_silu/asm/sass/fused_conv1d_silu_v3.cubin
```
