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
        s = alpha[b, t] * s + delta[b, t] * u[b, t]         # (H,)
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

---

## 3. 版本演进

### 3.1 v0：朴素基线（4 个分离 kernel）

**文件：** `fused_gated_delta_rule_v0.cu`

- 3 个 `LinearActivationKernel` 分别计算 decay/delta/state 投影，写 alpha/delta/u 到全局内存
- 1 个 `RecurrentDeltaRuleKernel` 读取中间 buffer 做循环更新
- 优点：结构简单清晰，验证正确性的基准
- 缺点：4 次 kernel launch 开销大；x_bt 被 H 个线程各自从全局读 H 次，严重冗余；中间 buffer 3 次写 + 3 次读浪费带宽

### 3.2 v1：融合投影（2 个 kernel）

**文件：** `fused_gated_delta_rule_v1.cu`

将 3 个投影 kernel 合并为 1 个 `FusedProjectionKernel`，每个线程在同一个循环中同时算 3 个点积：
- x 数据只读一次，3 个权重矩阵连续访问—改善缓存局部性
- kernel launch 从 4 次减到 2 次
- **实测效果：** 小尺寸稍有改善，大尺寸反而略慢于 v0，主要原因是从 3 个体积小的 kernel 变成了 1 个体积更大的 kernel，register pressure 升高但没有解决 x_bt 的 H 重读问题

### 3.3 v2：SMEM 缓存 x_bt + float4 向量化（关键优化）

**文件：** `fused_gated_delta_rule_v2.cu`

v2 找到了 v0/v1 的核心问题：**每个 (b,t) 位置的 x_bt 被 H 个线程同时从全局内存独立读取，总共产生 H× 的冗余全局访问。**

解决：用 shared memory 做 x_bt 的块内缓存。

```
grid(B, L), block(128), shared_mem = x[D × sizeof(float)]

每个 block 处理一个 (b,t):
  1. 协作式将 x_bt 从全局内存加载到 SMEM — 每个元素只读 1 次
  2. __syncthreads() 同步
  3. 每个线程从 SMEM 读 x，从全局读 W，
     用 float4 向量化计算 3 个点积 + 激活函数
  4. 写 alpha/delta/u 到全局内存
```

**额外优化：**
- **Float4 向量化访存：** W 从全局内存和 x 从 SMEM 都使用 float4（16 字节）加载，每 4 个 FMA 合并 1 次访存指令
- **SMEM 节省全局带宽：** H=256 时，SMEM 策略将 x 的全局读取量减少到 1/256

---

## 4. 性能对比

GPU 时间（ms），全部 PASS（与 CPU FP32 参考的 MaxAbsDiff < 1e-3）：

| 测试规模 (B,L,D,H) | **v0 (4 kernel)** | **v1 (2 kernel)** | **v2 (SMEM)** | **v2 vs v0** |
|:-----------------:|:----------------:|:----------------:|:-------------:|:-----------:|
| 1,128,64,32       | 0.0345           | 0.0267           | **0.0182**    | **1.9×** |
| 1,256,128,64      | 0.1174           | 0.1079           | **0.0343**    | **3.4×** |
| 2,512,256,128     | 1.0711           | 1.5072           | **0.6283**    | **1.7×** |
| 4,1024,512,256    | 16.705           | 21.798           | **10.010**    | **1.7×** |
| 8,2048,512,256    | 66.973           | 86.694           | **40.166**    | **1.7×** |

v2 在所有尺寸下均优于 v0，加速比稳定在 1.7~3.4×。v1 在中等以上尺寸反而比 v0 慢，说明单纯合并 kernel 而不解决访存冗余不足以带来收益。

---

## 5. Nsight Compute 性能分析

使用 `ncu --set basic` 对每个可执行文件的第一个 kernel launch 进行 profiling。
运行环境：NVIDIA RTX 5060 Ti (Blackwell sm_120) | CUDA 13.2 | Nsight Compute 2026.1.1

| 版本 | Kernel | Duration(us) | Compute% | MemBW% | L1% | L2% | Occupancy% | Reg/Thread | Block | Grid |
|---|---|---|---|---|---|---|---|---|---|---|
| fused_gated_delta_rule_v0 | 4 kernels (3×proj+1×recurrent) | 6.5 | 5.9% | 45.1% | 65.3% | 4.5% | 9.9% | 30 | 256 | 128 |
| fused_gated_delta_rule_v1 | FusedProjectionKernel | 13.0 | 5.7% | 65.4% | 79.5% | 6.3% | 8.4% | 39 | 256 | 128 |
| fused_gated_delta_rule_v2 | FusedProjectionSmemKernel | **2.5** | **43.3%** | **55.0%** | **33.8%** | **7.6%** | **10.4%** | 44 | 128 | 128 |

v2 的关键改进在 ncu 数据中清晰可见：
- **Duration 从 6.5us 降到 2.5us** — SMEM 缓存直接缩短了投影 kernel 的执行时间
- **MemBW% 从 45.1% 升到 55.0%** — SMEM 缓存减少了 x 的冗余全局读，剩余的内存带宽更多用于 W 和中间 buffer 的读写
- **L1% 从 65.3% 降到 33.8%** — x 数据通过 SMEM 访问（L1 是 global load/store 的通道），L1 压力减小
- **Compute% 从 5.9% 升到 43.3%** — 访存效率提高后，计算有更多机会执行

v1 的 Duration（13.0us）比 v0（6.5us）更长，这是因为 v1 的 FusedProjectionKernel 每个线程做 3 倍的点积工作量，但 grid 大小没变（同样是 128 blocks），kernel 执行时间自然变长。而 v2 通过 SMEM 缓存解决了访存冗余，投影 kernel 仅需 2.5us，实现真正的加速。

---

## 6. 产物路径

- **可执行文件：** `build/bin/fused_gated_delta_rule_v0`, `v1`, `v2`
- **结果 CSV：** `data/results/fused_gated_delta_rule_v{0,1,2}_results.csv`
- **量化评估可执行：** `build/bin/fused_gated_delta_rule_quant_eval`
- **量化评估 CSV：** `data/results/fused_gated_delta_rule_quant_eval.csv`
- **PTX/SASS：** `fused_gated_delta_rule/asm/ptx/`、`fused_gated_delta_rule/asm/sass/`
