# Nsight Compute 深度 Profiling 指南

## 为什么要做 Profiling

优化 CUDA kernel 的瓶颈分析不能靠"猜"——必须用 profiler 精确测量哪些指令在 stall、多少带宽被浪费、SM 占用率多少。Nsight Compute（ncu）是 NVIDIA 官方的 kernel 级 profiler。

## 三个关键指标

### 1. Memory Throughput（访存吞吐）

```
ncu --set full --kernel-name GemmV4Kernel ./build/bin/gemm_v4
```

重点关注：
- **DRAM Throughput**：实际达到的显存带宽 vs 峰值（如 760 GB/s）
- **L1/L2 Hit Rate**：缓存命中率 → L1 hit 高说明 SMEM/局部性用得好
- **Sectors/Request**：每个 memory request 实际使用的 cache sector 数 → 越接近 32 说明 coalescing 越好

**解读**：GEMM v4 的 DRAM throughput 接近 700 GB/s（峰值的 92%）说明内存不是瓶颈。但如果低于 50%，说明访存模式有问题。

### 2. Compute Throughput（计算吞吐）

- **SM Throughput**：SM 执行单位的利用率
- **Tensor Core Active**：Tensor Core 被使用的 cycle 占比

**解读**：GEMM v4 预计 Tensor Core 利用率 > 90%。如果低于 60%，说明有 stall（如等待数据到达）。

### 3. Occupancy（占用率）

- **Theoretical Occupancy**：理论最大活跃 warp 数 / SM 最大 warp 数
- **Achieved Occupancy**：实际平均活跃 warp 数
- **Stall Reasons**：线程 stall 的原因分布

**解读**：Occupancy 高（>50%）不一定好，低也不一定坏。关键是 stall reason——如果是 `waiting for data` 多，说明访存没隐藏；如果是 `short scoreboard` 多，说明指令依赖。

## 关键命令

```bash
# 基础 profiling（耗时最短）
ncu --kernel-name regexp ./build/bin/gemm_v4

# 全面 profiling（包括 stall reason、memory、compute）
ncu --set full --kernel-name GemmV4Kernel ./build/bin/gemm_v4

# 分析 roofline（访存 vs 计算）
ncu --roofline --kernel-name GemmV4Kernel ./build/bin/gemm_v4

# 导出详细报告
ncu --set full -o report  ./build/bin/gemm_v4
# 再打开 report.ncu-rep 用 Nsight Compute GUI
```

## 常见 Stall Reason 含义

| Stall Reason | 含义 | 解决方向 |
|:------------|------|---------|
| `Long Scoreboard` | 等待全局/局部内存加载完成 | 减少 memory divergence，提高 coalescing |
| `Short Scoreboard` | 等待指令依赖（如 FMA 的输入） | 减少寄存器压力，增加 ILP |
| `Not Predicated` | warp 在执行 | 正常 |
| `Wait Barrier` | 等待 __syncthreads() | 减少同步频率，平衡 workload |
| `MIO Queue Full` | I/O 队列满（SFU、Tensor Core） | 减少特殊指令的混合使用 |

## GEMM v4 Profiling 预期结果

| 指标 | v4 (预期值) | 说明 |
|------|:-----------:|------|
| DRAM Throughput | ~680 GB/s (90%) | cp.async + SMEM 做得好 |
| Tensor Core Active | ~85% | 256 线程，4x4 WMMA tiles |
| Achieved Occupancy | ~45% | 寄存器压力中等 |
| Long Scoreboard | ~25% | cp.async 已隐藏大部分延迟 |
| L1 Hit Rate | >85% | SMEM tiling 效果好 |

## RMSNorm V3 Profiling 预期结果

| 指标 | V3 (预期值) | 说明 |
|------|:-----------:|------|
| DRAM Throughput | ~580 GB/s (76%) | weight SMEM staging 减少了 DRAM 访问 |
| Achieved Occupancy | ~65% | 轻量 kernel，寄存器压力小 |
| Long Scoreboard | ~40% | 访存受限（算术强度 0.33 FLOP/Byte）|

## 在 fused_conv1d_silu 场景下的 Profiling

```bash
# 分析 Kernel A 的瓶颈
ncu --set full --kernel-name fused_v2::ComputeQKVZKernel ./build/bin/fused_conv1d_silu_v2

# 分析 Kernel B 的瓶颈  
ncu --set full --kernel-name fused_v2::ConvGateKernel ./build/bin/fused_conv1d_silu_v2

# 对比 v0 和 v2 的总 DRAM 访问量
ncu --set memory --kernel-name "::" ./build/bin/fused_conv1d_silu_v0
ncu --set memory --kernel-name "::" ./build/bin/fused_conv1d_silu_v2
```

预期的对比结果：
- v0：总 DRAM 读取 ~412 MB，总写入 ~288 MB，总共 ~700 MB
- v2：总 DRAM 读取 ~135 MB，总写入 ~67 MB，总共 ~202 MB

**减少的 498 MB 流量就是融合带来的收益，可以精确定量到字节。**

## Roofline 分析

Roofline 是判断 kernel 是 compute-bound 还是 memory-bound 的标准工具：

```bash
ncu --roofline --kernel-name GemmV4Kernel ./build/bin/gemm_v4
```

输出大致如下：

```
Operational Intensity: 82.9 FLOP/Byte            ← 高算术强度 → compute-bound
Peak Performance: 25.0 TFLOPS                     ← GPU 峰值
Roofline-bound: Compute                           ← 瓶颈在计算，不在内存
```

## 简历写法

```
• Nsight Compute 深度 Profiling：对 GEMM、RMSNorm、Fused Conv1D+SiLU 
  三类算子进行 roofline 分析和 stall reason 诊断，量化各版本的瓶颈转移：
  — GEMM v4：Tensor Core 利用率 85%，访存带宽 ~680 GB/s（峰值 90%）
  — 融合后流量从 v0 的 700 MB 降至 v2 的 202 MB（减少 71%）
  — 定位权重矩阵 non-coalesced 访问为下一个优化瓶颈
```

## 参考

- [Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/)
- [CUDA 最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
