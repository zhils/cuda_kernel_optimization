# Flash Attention CUDA 实现

## 项目目标

实现标准的 Attention 机制（Q @ K^T → softmax → @ V），对比朴素实现（v0，显式物化 N×N 注意力矩阵）与 Flash Attention（v1，tiled online softmax 不物化 N² 矩阵）的性能差异与正确性。

## 数学定义

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{D}}\right)V
$$

纯文本：`Attention(Q,K,V) = softmax((Q @ K^T) / sqrt(D)) @ V`。

$$
Q,K,V \in \mathbb{R}^{B\times H\times N\times D}
$$

纯文本：`Q,K,V` 的形状均为 `(B,H,N,D)`，`softmax` 作用在最后一维。

**朴素实现（v0）：** 显式计算并存储 `S = Q @ K^T`（形状 `(B,H,N,N)`），再 softmax 得 `P`（形状 `(B,H,N,N)`），再 `P @ V -> O`。内存 `O(N^2)`，带宽 `O(N^2 * D)`。

**Flash Attention（v1）：** 将 Q、K、V 沿 N 维分块（Br=32，Bc=32），每个 tile 内计算局部 S 并立即通过 online softmax 累加到 O 的寄存器累加器中。内存 O(Br·Bc + (Br+2·Bc)·D)，带宽 O(N·D × N/Bc)——不物化 N² 矩阵。

## Online Softmax 的核心公式

对于每一行，维护运行最大值 m 和运行和 ℓ：

```
处理第 j 个 KV tile 后:
  S_j = Q_row @ K_tile_j^T           ← tile 注意力分数
  m_new = max(m, max(S_j))            ← 更新全局最大值
  ℓ = ℓ * exp(m - m_new) + sum(exp(S_j - m_new))   ← 更新归一化常数
  O = O * exp(m - m_new) + exp(S_j - m_new) @ V_j  ← 重新缩放并累加
  m = m_new

所有 tile 处理完后:
  O_final = O / ℓ                     ← 最终归一化
```

**关键洞察：** 当新的 tile 到来时，之前累加的部分需要按 exp(m_old - m_new) 重新缩放。这保证了即使最大值在后续 tile 中更新，之前的结果也自动适配。最终的除法 O/ℓ 相当于完整 softmax。

## 版本演进

### v0：朴素基线（显式 N² 矩阵）

- 3 个 kernel：ComputeScores + SoftmaxScores + ApplyValues
- 显式存储 `S/P`（形状 `(B,H,N,N)`）→ 内存 `O(N^2)`，大 N 时不可行
- 用于验证正确性和衡量 Flash Attention 的加速比

### v1：Tiled Online-Softmax（Flash Attention）

- 单 kernel，grid(B×H) × block(256)
- Q tile (Br×D) + K tile (Bc×D) + V tile (Bc×D) 加载到共享内存
- O 在寄存器中跨所有 KV tile 累加，仅最后写回全局内存
- Warp shuffle 做 D 维度的归约

### v2：Fallback kernel（WGMMA 不可用时的可移植版本）

- 2D grid：grid(B×H, N)，每线程处理一行
- 串行 dot product + softmax，无 tile 分块

### v3：访存效率优化（2D Grid + 消除外层 Q tile 循环）

**核心问题：** v1 使用 1D grid `(B*H)`，B=1,H=1 时仅 1 个 block，SM Occupancy 仅 16.67%。每个 block 要在外层循环处理所有 Q tile，导致 Q 被反复从全局内存加载。

**v3 优化：**

| 优化 | 说明 |
|------|------|
| **2D grid** | `grid(B*H, ceil(N/Br))`，每 block 只处理一个 Q tile，block 数从 `B*H` 增至 `B*H * N/Br`，SM 占用率大幅提升 |
| **消除外层循环** | 每个 block 不再循环 Q tile，Q_smem 仅加载 1 次，寄存器使用降低 |
| **float4 向量化** | SMEM 加载使用 `float4`，全局内存指令数降至 1/4 |
| **精简同步** | 每次 K/V tile 仅 2 次 `__syncthreads()`（加载后 + 计算后） |

**预期效果：** block 数量从 1~8 增至 32~256+，SM Occupancy 从 16.67% → ~60%+；消除了 Q 的重复全局读取。

### v4：计算强度优化（Bank-free SMEM + ILP + FMA）

**核心问题：** v1/v3 的 SMEM 行跨度 D 恰好等于 32/64/128（32 的倍数），导致 32-bank 冲突严重——同 warp 的 32 个线程若同时访问不同行同一列，将命中同一 bank。

**v4 优化：**

| 优化 | 说明 |
|------|------|
| **SMEM padding** | `D_pitch = D + 1`，插入 1 个浮点 padding 破坏 bank 对齐模式 |
| **`__fmaf_rn`** | `sum = a*b + sum` → `sum = __fmaf_rn(a, b, sum)`，一条指令完成乘加 |
| **双路 ILP** | 两个独立累加器 `sum0`/`sum1` 交替 FMA，隐藏 FMA 延迟（~4 cycle） |
| **寄存器缓存 Q** | 从 SMEM 读取的 Q 值存入寄存器 `q_reg[ri][k]`，减少 SMEM 读取 |
| **`#pragma unroll`** | 强制展开内层 dot product 循环，增加指令级并行 |

| 版本 | Grid | Q 加载 | SMEM 布局 | 归约方式 | 指令优化 | 预期 Occupancy |
|:----|:----:|:------:|:---------:|:--------:|:--------:|:-------------:|
| v0 | (B*H) | 无 SMEM | 无 | 无 | 无 | 33% |
| v1 | (B*H) | 每 Q tile | 密集(D) | Warp Shuffle | 无 | 17% |
| v2 | (B*H, N) | 无 SMEM | 无 | 无 | 无 | 27% |
| v3 | (B*H, N/Br) | 1 次 | 密集(D) | Warp Shuffle | float4 loads | ~60% |
| v4 | (B*H, N/Br) | 1 次 | **padded(D+1)** | Warp Shuffle | **FMA + ILP + unroll** | ~60% |

## 性能数据

| B | H | N | D | v0 (ms) | v1 (ms) | 加速比 |
|:-:|:-:|:-:|:-:|:-------:|:-------:|:-----:|
| 1 | 1 | 64 | 32 | 0.238 | 0.147 | 1.6x |
| 1 | 1 | 128 | 64 | 1.508 | 0.842 | 1.8x |
| 1 | 2 | 256 | 64 | 5.734 | 3.334 | 1.7x |
| 1 | 4 | 512 | 64 | 20.867 | 13.388 | 1.6x |
| 1 | 8 | 1024 | 32 | 49.096 | 36.393 | 1.3x |

加速比在中小规模最为明显（节省 N² 矩阵的写回和重新读取），大规模时计算（matmul）占主导，加速比较稳定在 1.3-1.6x。

## Nsight Compute 瓶颈分析（2026-05-09）

命令：`ncu --set basic --target-processes all --kernel-name-base demangled`。  
统计口径：每个可执行文件取 Duration 最大的一次 kernel launch。

| 目标 | Max Duration(us) | Compute(SM) | DRAM | Memory | Achieved Occupancy | Reg/Thr | 结论 |
|:-----|-----------------:|------------:|-----:|-------:|-------------------:|--------:|:-----|
| `flash_attention_v0` | 599.97 | 0.69% | 2.46% | 11.53% | 33.04% | 40 | 多 kernel 分离路径，整体利用率低 |
| `flash_attention_v1` | 705.70 | 0.81% | 0.19% | 0.81% | 16.67% | 72 | 当前 launch 网格太小（`Waves/SM≈0.01`），严重欠并行 |
| `flash_attention_v2` | 551.23 | 65.09% | 0.08% | 65.09% | 27.01% | 37 | 当前为 fallback v2 内核，具备可分析的中等算力利用 |
| `flash_attention_v3` | 6300.00¹ | 36.61% | 0.95% | 36.61% | 33.32% | 96 | 2D Grid 消除外层 Q 循环，grid 并行度由 N/Br 决定 |
| `flash_attention_v4` | 4610.00¹ | 50.86% | 2.15% | 50.86% | 44.88% | 72 | Bank-free SMEM + ILP + FMA，算力利用进一步提升 |

补充：
- 由于当前工具链无法汇编 WGMMA PTX，`flash_attention_v2` 暂时使用可移植 fallback kernel 保持可构建/可分析。
- 原始报告：`data/ncu_reports/text/flash_attention_v0.txt`、`flash_attention_v1.txt`、`flash_attention_v2.txt`。
- ¹ v3/v4 的 ncu 数据来自 `data/ncu_reports/manual_fill/flash_attention_v3.txt` 与 `flash_attention_v4.txt`（按各可执行文件 Duration 最大 launch 统计）。
- v3/v4 的大规模 profiling 可在本地运行 `NCU_QUICK=1 ncu --set basic ./build/bin/flash_attention_v3` 获得。

---

## Warp Stall 原因分析

| 版本 | #1 Stall | #2 Stall | #3 Stall | #4 Stall | #5 Stall |
|:----|:---------|:---------|:---------|:---------|:---------|
| v0 | Short Scoreboard 38.9% | Long Scoreboard 31.7% | Wait 26.9% | No Instruction 1.2% | Not Selected 1.0% |
| v1 | Long Scoreboard 52.7% | Not Selected 27.1% | Wait 18.3% | Short Scoreboard 1.2% | No Instruction 0.6% |
| v2 | Long Scoreboard 52.7% | Not Selected 27.1% | Wait 18.3% | Short Scoreboard 1.2% | No Instruction 0.6% |
| v3 | Short Scoreboard 40.7% | Wait 33.8% | No Instruction 15.0% | Not Selected 6.2% | Long Scoreboard 2.3% |
| v4 | Long Scoreboard 47.2% | Short Scoreboard 31.6% | Wait 16.7% | Not Selected 3.7% | Math Pipe Throttle 0.3% |

v0/v3 的 Short Scoreboard 较高，说明小规模测试时数据主要驻留在 L1/L2 缓存中，等待缓存加载而非全局内存。v1/v2/v4 以 Long Scoreboard 为主，符合大型 tile 多次加载 K/V 时受全局内存延迟限制的预期。

## 构建

```bash
cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=120
make flash_attention_v0 flash_attention_v1 flash_attention_v2 flash_attention_v3 flash_attention_v4 -j$(nproc)
cd ..

./build/bin/flash_attention_v0  # 朴素基线
./build/bin/flash_attention_v1  # Tiled Flash Attention
./build/bin/flash_attention_v2  # Fallback kernel
./build/bin/flash_attention_v3  # 2D Grid + float4
./build/bin/flash_attention_v4  # Bank-free SMEM + ILP + FMA
```

## 原理参考

- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)
- Online softmax: "Online normalizer calculation for softmax" (Milakov & Gimelshein, 2018)

## 主场景性能口径（统一）

主指标统一为主场景 `gpu_ms`，NCU 吞吐仅用于瓶颈归因。

| 实现 | 主场景维度 | GPU耗时(ms) | 校验状态 | 数据文件 |
|---|---|---:|---|---|
| `flash_attention_v4` | `B=1,H=8,N=1024,D=32` | 4.23957 | PASS | `data/results/flash_attention_v4_results.csv` |

环境口径：`RTX 5060 Ti (sm_120) + CUDA 13.2`。
统一汇总：`data/results/main_scenario_unified.csv`（retest tag: `20260512_manual_retest`）。

## 已知边界与后续补充

- 当前 `v2` 为可移植 fallback 路径，不代表最终 Tensor Core/WGMMA 上限。
- 建议补充 `N=2048/4096` 的同口径 NCU 与主场景表，形成大序列趋势曲线。

