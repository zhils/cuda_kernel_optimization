# Flash Attention CUDA 实现

## 项目目标

实现标准的 Attention 机制（Q @ K^T → softmax → @ V），对比朴素实现（v0，显式物化 N×N 注意力矩阵）与 Flash Attention（v1，tiled online softmax 不物化 N² 矩阵）的性能差异与正确性。

## 数学定义

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(D)) @ V

其中 Q, K, V ∈ R^{B×H×N×D}，softmax 作用于最后一维
```

**朴素实现（v0）：** 显式计算并存储 S = Q @ K^T ∈ R^{B×H×N×N}，再 softmax → P ∈ R^{B×H×N×N}，再 P @ V → O。内存 O(N²)，带宽 O(N²·D)。

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
- 显式存储 S(P) ∈ R^{B×H×N×N} → 内存 O(N²)，大 N 时不可行
- 用于验证正确性和衡量 Flash Attention 的加速比

### v1：Tiled Online-Softmax（Flash Attention）

- 单 kernel，grid(B×H) × block(256)
- Q tile (Br×D) + K tile (Bc×D) + V tile (Bc×D) 加载到共享内存
- O 在寄存器中跨所有 KV tile 累加，仅最后写回全局内存
- Warp shuffle 做 D 维度的归约

## 性能数据

| B | H | N | D | v0 (ms) | v1 (ms) | 加速比 |
|:-:|:-:|:-:|:-:|:-------:|:-------:|:-----:|
| 1 | 1 | 64 | 32 | 0.238 | 0.147 | 1.6x |
| 1 | 1 | 128 | 64 | 1.508 | 0.842 | 1.8x |
| 1 | 2 | 256 | 64 | 5.734 | 3.334 | 1.7x |
| 1 | 4 | 512 | 64 | 20.867 | 13.388 | 1.6x |
| 1 | 8 | 1024 | 32 | 49.096 | 36.393 | 1.3x |

加速比在中小规模最为明显（节省 N² 矩阵的写回和重新读取），大规模时计算（matmul）占主导，加速比较稳定在 1.3-1.6x。

## 原理参考

- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)
- Online softmax: "Online normalizer calculation for softmax" (Milakov & Gimelshein, 2018)

## Nsight Compute 性能分析


使用 `ncu --set basic` 对每个可执行文件的第一个 kernel launch 进行 profiling。
运行环境：NVIDIA RTX 5060 Ti (Blackwell sm_120) | CUDA 13.2 | Nsight Compute 2026.1.1

| 版本 | Kernel | Duration(us) | Compute% | MemBW% | L1% | L2% | Occupancy% | Reg/Thread | Block | Grid |
|---|---|---|---|---|---|---|---|---|---|---|
| flash_attention_v0 | FlashAttnNaiveKernel | 122.8 | 0.1% | 1.2% | 44.1% | 0.2% | 4.2% | 36 | 64 | 1 |
| flash_attention_v1 | FlashAttnTiledKernel | 148.9 | 0.6% | 0.9% | 20.1% | 0.1% | 16.7% | 96 | 256 | 1 |
**说明：** ncu `--set basic` 默认对程序的**第一个 kernel launch** 进行 profiling。对于 GEMM 等算子，这对应最小测试尺寸（128×128），GPU 远未饱和。因此表格中的 Compute% / MemBW% 表示的是**小尺寸下的资源利用率**，用于横向对比各版本的寄存器压力、occupancy 等结构性差异。大尺寸下的实际性能请参考各算子 README 中的完整 benchmark 表格。

