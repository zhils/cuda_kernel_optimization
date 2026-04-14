# RMSNorm

## 数学定义

```
y = x / sqrt(mean(x²) + ε) × γ

其中 mean(x²) = (1/C) × Σ_{i=0}^{C-1} x_i²
```

RMSNorm（Root Mean Square Layer Normalization）在现代 LLM（如 LLaMA、Qwen 等）中被广泛使用。它是 LayerNorm 的更简洁、更高效替代方案：去掉均值中心化步骤，只计算 RMS 来做归一化。

## 理论性能分析

### 算术强度

| 指标 | 公式 | 数值 |
|------|------|------|
| 数据搬运量 | 读 x + 读 γ + 写 y = (2R×C + C) × 4 bytes | ~8RC bytes |
| 计算量 | R × (C 次乘法 + C 次加法 + 1 次 rsqrt + C 次乘法 + C 次乘法) | ~4RC FLOPs |
| 算术强度 | 4RC / 8RC | **~0.5 FLOP/Byte** |

### Roofline 分类

与 Softmax 类似，RMSNorm 也是典型 **内存受限** 算子。优化目标是最大化 DRAM 带宽利用率。

### 理论峰值

- RTX 5060 Ti DRAM 带宽：~XX GB/s
- 最小数据搬运量：读 x 一次 + 读 γ 一次 + 写 y 一次
- 对于形状 (512, 4096)：(512 × 4096 × 2 + 4096) × 4 bytes ≈ 16 MB → 理论最短时间 = XX μs

## 优化版本

### V0：朴素实现（每行单线程）

**文件：** `rmsnorm_v0_naive.cu`

每个线程处理整行：先串行计算 sq_sum，再串行做归一化。

- **问题：** 行内无并行，SM 利用率极低
- **问题：** 每个线程要读取整行两次（一次算 sq_sum，一次写输出）
- **Nsight 诊断：** 大量 warp 空闲，内存带宽严重未被利用

### V1：Warp 级 Shuffle 归约

**文件：** `rmsnorm_v1_warp_reduce.cu`

每行一个 block。多个线程利用 `__shfl_sync` 的 warp 归约协同计算 sq_sum。

- **核心思路：** 在 warp 内并行归约 sq_sum
- **广播：** 计算出的 rms 通过共享内存共享给其它线程，然后全部线程共同写输出
- **局限：** 当 blockDim > 32 时，仅一个 warp 参与归约贡献

### V2：向量化内存访问（float4）

**文件：** `rmsnorm_v2_vectorized.cu`

- 使用 `float4` 向量化读取计算 sq_sum：128-bit 事务替代 32-bit
- 通过共享内存 `warp_sums[]` 数组完成跨 warp 归约
- 使用 `float4` 向量化写回输出
- 所有线程都参与归约和输出写入

### V3：融合 Kernel

**文件：** `rmsnorm_v3_fused.cu`

将所有优化合并到一个融合 kernel：

- 向量化 `float4` 读写
- 完整的跨 warp 归约（warp shuffle + 共享内存）
- 使用 `__fmul_rn` 内建函数保证可重复的舍入行为
- 归约和输出两个阶段均保持全线程活跃
- 面向 LLM 常见 hidden dim（4096、8192）优化

## 性能结果

<!-- TODO: 运行 rmsnorm_benchmark_all 后补充 -->

| Rows | Cols | Naive (ms) | WarpReduce (ms) | Vectorized (ms) | Fused (ms) | cuDNN (ms) |
|------|------|------------|-----------------|-----------------|------------|------------|
| 32 | 32 | | | | | |
| 128 | 256 | | | | | |
| 1111 | 222 | | | | | |
| 4096 | 4096 | | | | | |

## NVIDIA 参考 API

- **cuDNN：** `cudnnNormalizationForward`（`CUDNN_NORM_RMS`）
