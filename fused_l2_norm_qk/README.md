# Fused L2 Norm Q/K

## 目标

将 Query 和 Key 的 L2 归一化合并为单个 CUDA kernel，减少中间张量读写。

---

## 数学定义

对张量 $X \in \mathbb{R}^{(B, N, H)}$ 每行做 L2 归一化：

$$
\|x_{b,n}\|_2 = \sqrt{\sum_{h=0}^{H-1} x_{b,n,h}^2}
$$

$$
\hat{x}_{b,n,h} = \frac{x_{b,n,h}}{\|x_{b,n}\|_2 + \epsilon}
$$

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

| 内核 | Duration(us) | Compute(SM) | DRAM | Achieved Occupancy | Reg/Thr |
|:----|:-----------:|:-----------:|:----:|:------------------:|:-------:|
| `L2NormKernel` | 402.53 | 29.08% | 25.20% | 89.22% | 22 |

Occupancy 很高，算存较均衡，受归约与访存共同限制。v1/v2 通过减少同步与 ILP 优化提升计算占比。

---

## 构建

```bash
cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=120 && make fused_l2_norm_qk_v0 -j$(nproc)
cd ..
./build/bin/fused_l2_norm_qk_v0

# v1/v2 同理
./build/bin/fused_l2_norm_qk_v1
./build/bin/fused_l2_norm_qk_v2
```
