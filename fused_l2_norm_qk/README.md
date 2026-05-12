# Fused L2 Norm Q/K

## 目标

将 Query 和 Key 的 L2 归一化合并为单个 CUDA kernel，减少中间张量读写。

---

## 数学定义

对张量 `X`（形状 `(B,N,H)`）每行做 L2 归一化：

$$
\|x_{b,n}\|_2 = \sqrt{\sum_{h=0}^{H-1} x_{b,n,h}^2}
$$
纯文本：`norm2(x[b,n]) = sqrt(sum_{h=0..H-1}(x[b,n,h]^2))`。

$$
\hat{x}_{b,n,h} = \frac{x_{b,n,h}}{\|x_{b,n}\|_2 + \epsilon}
$$
纯文本：`x_hat[b,n,h] = x[b,n,h] / (norm2(x[b,n]) + eps)`。

---

## 版本演进

### v0：基线（Q 和 K 分离）

Q 和 K 的 L2 归一化分别实现为独立 CUDA kernel。

### v1：访存效率优化（Q/K 融合 + float4 + Warp Shuffle）

将 Q/K 两个 kernel 融合为 **单个 3D grid kernel**：

- 3D grid：`(B, max(N_q,N_k), 2)`，`blockIdx.z=0` 处理 Q，`=1` 处理 K
- 消除 1 次 kernel launch 开销（launch latency + 调度开销）
- **Warp Shuffle** 替代 SMEM 树归约，消除 `__syncthreads()` 屏障延迟
- 寄存器直接加载 val + 平方累加，避免 SMEM staging round-trip
- 跨 warp 归约仅需 1 个 `__syncthreads()`

**预期效果：** 减少 kernel launch 和同步开销，提升访存带宽利用率。

### v2：计算强度优化（2 行/block + ILP + FMA）

在 v1 基础上，每 block 处理 **2 行**，核心思路是提升计算占比：

- **4 路 ILP**：每线程 4 个独立累加器 `sum[4]`，编译器可跨累加器调度 FMA
- **`__fmaf_rn`**：融合乘加指令 `a = a + b*b` 用 `__fmaf_rn(b, b, a)`，指令数减半
- **`#pragma unroll`**：展开内层循环，暴露 ILP
- 归约阶段合并 4 路累加器，归约开销占比从 ~20% 降至 ~5%
- 第 2 次访存（写回）复用 4 路 ILP 模式，掩盖写延迟

| 版本 | 策略 | Kernel 数 | 归约方式 | 每 block 行数 | ILP | 指令优化 |
|:----|------|:---------:|:--------:|:------------:|:---:|:--------:|
| v0 | Q/K 分离 | 2 | SMEM 树归约 | 1 | 1 | 无 |
| v1 | 融合 + Warp Shuffle | 1 (3D grid) | Warp Shuffle | 1 | 1 | float4 |
| v2 | 2 行/block + ILP + FMA | 1 (3D grid) | Warp Shuffle | **2** | **4** | **`__fmaf_rn` + unroll** |

---

## Nsight Compute 瓶颈分析

`ncu --set basic`，`fused_l2_norm_qk_v0`：

| 内核 | Duration(us) | Compute(SM) | DRAM | Memory | Achieved Occupancy | Reg/Thr |
|:----|:-----------:|:-----------:|:----:|:------:|:------------------:|:-------:|
| `L2NormKernel` (v0) | 402.53 | 29.08% | 25.20% | 29.08% | 89.22% | 22 |
| `FusedL2NormKernel` (v1) | 225.24 | 20.76% | 4.31% | 11.81% | 67.13%¹ | 36 |
| `FusedL2NormV2Kernel` (v2) | 238.10 | 12.96% | 5.34% | 10.59% | 55.21%¹ | 48 |

¹ v1/v2 Occupancy 为小规模测试数据，大规模时更高（v1 最高 91%，v2 最高 77%）。

Occupancy 很高，算存较均衡，受归约与访存共同限制。v1/v2 通过减少同步与 ILP 优化提升计算占比。

---

## Warp Stall 原因分析

| 版本 | #1 Stall | #2 Stall | #3 Stall | #4 Stall | #5 Stall |
|:----|:---------|:---------|:---------|:---------|:---------|
| v0 | Long Scoreboard 38.1% | Short Scoreboard 26.4% | Wait 20.7% | Not Selected 7.0% | Mio Throttle 5.2% |
| v1 | Long Scoreboard 46.1% | Wait 24.2% | Short Scoreboard 17.0% | Not Selected 9.0% | Math Pipe Throttle 1.5% |
| v2 | Long Scoreboard 35.4% | Short Scoreboard 27.9% | Wait 25.0% | No Instruction 5.6% | Not Selected 5.5% |

三个版本的 stall 分布较为均衡，Long Scoreboard、Short Scoreboard、Wait 三者合计占比约 85-90%，说明 L2 Norm 的瓶颈是混合的：加载 x 行的全局内存延迟（Long Scoreboard）、缓存访问（Short Scoreboard）以及 warp shuffle 归约的同步等待（Wait）共同构成主要延迟来源。

## 构建

```bash
cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=120 && make fused_l2_norm_qk_v0 -j$(nproc)
cd ..
./build/bin/fused_l2_norm_qk_v0

# v1/v2 同理
./build/bin/fused_l2_norm_qk_v1
./build/bin/fused_l2_norm_qk_v2
```

---

## 主场景性能口径（统一）

主指标统一为主场景 `gpu_ms`，NCU 吞吐仅用于瓶颈归因。

| 实现 | 主场景维度 | GPU耗时(ms) | 校验状态 | 数据文件 |
|---|---|---:|---|---|
| `fused_l2_norm_qk_v2` | `B=8,N_q=2048,H_q=256,N_k=2048,H_k=256` | 0.19774 | PASS | `data/results/fused_l2_norm_qk_v2_results.csv` |

环境口径：`RTX 5060 Ti (sm_120) + CUDA 13.2`。
统一汇总：`data/results/main_scenario_unified.csv`（retest tag: `20260512_manual_retest`）。

## 已知边界与后续补充

- 当前主场景为 `N_q=N_k`，尚未系统覆盖非对称长度（`N_q != N_k`）。
- 建议补充更大 `H`（如 512）及跨 batch 的吞吐扩展曲线。
