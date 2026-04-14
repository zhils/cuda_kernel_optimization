# Softmax

## 数学定义

对长度为 C 的每一行向量 x：

```
softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

数值稳定版本通过减去 `max(x)` 来避免 `exp()` 溢出。

## 理论性能分析

### 算术强度

| 指标 | 公式 | 数值 |
|------|------|------|
| 数据搬运量 | 读输入 + 写输出 = 2 × R × C × 4 bytes | 8RC bytes |
| 计算量 | R × (C 次比较 + C 次 exp + C 次加法 + C 次除法) | ~4RC FLOPs |
| 算术强度 | 4RC / 8RC | **0.5 FLOP/Byte** |

### Roofline 分类

算术强度约为 0.5 FLOP/Byte，Softmax 在各种输入规模下都属于典型 **内存受限**。
优化重点是提升 DRAM 带宽利用率并减少内存访问轮次。

### 理论峰值

- RTX 5060 Ti DRAM 带宽：~XX GB/s
- 最小数据搬运量（online softmax）：读一次 + 写一次 = 2 × R × C × 4 bytes
- 512×4096 的理论最短时间：(2 × 512 × 4096 × 4) / (XX × 10⁹) = XX μs

## 优化版本

### V0：朴素实现（每行单线程）

**文件：** `softmax_v0_naive.cu`

每个线程顺序处理整行：3 次遍历（max → exp+sum → normalize）。

- **问题：** 行内无并行，对大 C 场景浪费 GPU 资源
- **问题：** 每行需要 3 次全局内存遍历
- **Nsight 诊断：** SM 利用率很低，内存带宽未充分利用

### V1：共享内存 Block 归约

**文件：** `softmax_v1_shared_mem.cu`

每行使用一个 block，线程通过共享内存树形归约协同计算 max 和 sum。

- **核心思路：** 并行归约将 O(C) 串行工作转为 O(C/T + log T) 并行工作
- **仍是 3 次遍历：** max 归约 → exp+sum → normalize
- **共享内存用量：** 归约工作区为 `blockDim.x × sizeof(float)`

### V2：Online Softmax（单次累积）

**文件：** `softmax_v2_online.cu`

实现在线归一化算法（Milakov & Gimelshein, 2018）：同时维护运行中的 max 和 sum，当发现新最大值时对部分和进行重缩放。

- **核心思路：** 内存遍历从 3 次降到 2 次（一次累积 + 一次输出）
- **数值稳定性：** 通过在线重缩放保证：`sum = sum × exp(old_max - new_max) + exp(val - new_max)`
- **Block 归约：** `(max, sum)` 成对归约，并进行正确重缩放

### V3：Warp Shuffle + 向量化读取

**文件：** `softmax_v3_warp_shuffle.cu`

针对可放入单个 warp 的行（cols ≤ 128）：

- 使用 `__shfl_down_sync` 做 warp 内 max/sum 归约 —— **零共享内存**
- `float4` 向量化读取：内存事务数减少 4×
- 使用 `__shfl_sync(..., 0)` 将最终 max/sum 广播到所有 lane
- 由于不使用共享内存，可获得更高占用率

### cuDNN 参考实现

**文件：** `softmax_cudnn_ref.cu`

`cudnnSoftmaxForward`（`CUDNN_SOFTMAX_ACCURATE` 模式）—— NVIDIA 的优化实现。

## 性能结果

<!-- TODO: 运行 softmax_benchmark_all 后补充 -->

| Rows | Cols | Naive (ms) | SharedMem (ms) | Online (ms) | WarpShuffle (ms) | cuDNN (ms) |
|------|------|------------|----------------|-------------|------------------|------------|
| 64 | 512 | | | | | |
| 128 | 1024 | | | | | |
| 256 | 4096 | | | | | |
| 512 | 4096 | | | | | |

## NVIDIA 参考 API

- **cuDNN：** `cudnnSoftmaxForward` / `cudnnSoftmaxBackward`
