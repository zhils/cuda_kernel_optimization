# Nsight Compute Kernel 性能分析标准流程

> 本文档回答：用 Nsight Compute 分析 kernel 性能，应该遵循什么流程？看哪些核心指标？

---

## 1. Nsight Compute 快速入门

### 1.1 安装确认

```bash
# 确认已安装（通常随 CUDA Toolkit 一起）
which ncu
ncu --version

# 如果没安装
sudo apt install nsight-compute
```

### 1.2 最基本用法

```bash
# 分析一个可执行文件中的某个 kernel
ncu --kernel-name GemmV3Kernel ./build/bin/gemm_v3

# 输出到文件（推荐）
ncu --kernel-name GemmV3Kernel -o gemm_v3_report ./build/bin/gemm_v3

# 打开 GUI 查看报告
ncu-ui gemm_v3_report.ncu-rep
```

---

## 2. 标准分析流程（四步法）

### Step 1: 快速扫描 —— 30 秒定位问题

```bash
ncu --set full --kernel-name YourKernelName ./your_executable
```

**这 30 秒你要看什么？**

| 指标 | 正常范围 | 如果异常说明什么 |
|------|----------|-----------------|
| **Duration** | 取决于 kernel | 和上次 benchmark 对比 |
| **Compute Throughput** | > 60% for compute-bound | < 30% → SM 空转，查看 stall reasons |
| **Memory Throughput** | > 60% for memory-bound | < 30% → 没吃满带宽，查看访存模式 |
| **Registers/Thread** | < 128 for good occupancy | > 128 → occupancy 可能低于 25% |
| **Occupancy** | > 50% (Theoretical) | < 25% → 严重受限，可能是 regs/SMEM 太大 |

### Step 2: 深入分析 —— 根据瓶颈类型选择不同 metrics

#### A. 如果 kernel 是 memory-bound

```bash
# 聚焦访存行为
ncu --kernel-name YourKernel \
    --metrics \
    dram__bytes.sum,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
    lts__throughput.avg.pct_of_peak_sustained_elapsed,\
    smsp__inst_executed.avg.pct_of_peak_sustained_elapsed \
    ./your_executable
```

| 指标 | 含义 | 优化方向 |
|------|------|----------|
| `dram__throughput...` | DRAM 带宽利用率 | < 50% → 访存不是瓶颈（或者访存模式不好） |
| `l1tex__throughput...` | L1 cache 吞吐 | 高 → L1 命中好；低 → 考虑 __ldg |
| `lts__throughput...` | L2 cache 吞吐 | 高 → 数据复用好；低 → 考虑 tiling |
| `smsp__inst_executed...` | 指令执行占比 | memory-bound kernel 这应该 < 20%（大部分时间在等数据） |

#### B. 如果 kernel 是 compute-bound

```bash
# 聚焦计算行为
ncu --kernel-name YourKernel \
    --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    smsp__warps_active.avg.pct_of_peak_sustained_elapsed,\
    smsp__inst_executed.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_launched.avg,\
    sm__maximum_warps_per_active_cycle.avg \
    ./your_executable
```

| 指标 | 含义 | 优化方向 |
|------|------|----------|
| `sm__throughput...` | SM 利用率 | 你的 V4 在 4096³ 应该看到 ~50% |
| `smsp__warps_active...` | 活跃 warp 占比 | < 50% → 大量 warp stall |
| `sm__warps_launched` | 总 warp 数 | 太少 → 增大 grid |
| `sm__maximum_warps_per_active_cycle` | 理论最大活跃 warp | 用来算 actual occupancy |

### Step 3: Stall 分析 —— 找到"线程在等什么"

```bash
# 这是最重要的一步！
ncu --kernel-name YourKernel \
    --metrics \
    smsp__average_warps_issue_stalled_barrier_per_cycle_active.ratio,\
    smsp__average_warps_issue_stalled_dispatch_stall_per_cycle_active.ratio,\
    smsp__average_warps_issue_stalled_imc_miss_per_cycle_active.ratio,\
    smsp__average_warps_issue_stalled_lg_throttle_per_cycle_active.ratio,\
    smsp__average_warps_issue_stalled_long_scoreboard_per_cycle_active.ratio,\
    smsp__average_warps_issue_stalled_math_pipe_throttle_per_cycle_active.ratio,\
    smsp__average_warps_issue_stalled_membar_per_cycle_active.ratio,\
    smsp__average_warps_issue_stalled_not_selected_per_cycle_active.ratio,\
    smsp__average_warps_issue_stalled_short_scoreboard_per_cycle_active.ratio,\
    smsp__average_warps_issue_stalled_sleeping_per_cycle_active.ratio,\
    smsp__average_warps_issue_stalled_wait_per_cycle_active.ratio \
    ./your_executable
```

**Stall 原因解读表：**

| Stall 原因 | 含义 | 常见场景 | 解法 |
|-----------|------|----------|------|
| **Long Scoreboard** | 等待全局内存/L2/L1 数据到达 | Memory-bound kernel 的主因 | 减少 global access；增大 tile |
| **Short Scoreboard** | 等待共享内存数据或上一个指令结果 | SMEM bank conflict 或指令依赖 | 检查 bank conflict；增加指令并行度 |
| **Not Selected** | 有其他 warp 在跑，当前 warp 没被调度到 | **这是好事！** 说明 SM 被充分利用 | 不需要处理 |
| **Barrier** | `__syncthreads` 等待 | 同步开销；线程间负载不均衡 | 减少 syncthreads；warp-level 操作替代 |
| **Membar** | 等待内存 fence | cp.async wait_group 或 __threadfence | 让异步操作更早提交 |
| **LG Throttle** | Load/Store 单元满了（太多 in-flight request） | 内存请求太多 | 减少并发 load；用 prefetch |
| **Math Pipe Throttle** | 计算单元被占满 | Tensor Core / FMA 太密集 | **也是好事**：说明计算资源被用满了 |
| **Sleeping** | 线程还没被分配到 | grid 太小，block 数不够 | 增大 grid |
| **IMC Miss** | Immediate constant cache miss | 很少见 | 一般不需要关注 |

**你的 V4 在 4096³ 上预期的 stall 分布：**

```
Long Scoreboard:  30-40%  ← 等全局内存数据（WMMA 加载 fragment 需要等待）
Not Selected:     25-30%  ← 其他 warp 在跑（说明 SM 有多个 warp 可以调度）
Short Scoreboard: 10-15%  ← 等上一个 mma 的结果（Tensor Core 延迟 ~4 cycles）
Math Pipe:        5-10%   ← Tensor Core 被占满（好信号！）
Barrier:          10-15%  ← __syncthreads（cp.async 双缓冲的缓冲交换）
```

### Step 4: 共享内存分析

```bash
# 检查 bank conflict
ncu --kernel-name YourKernel \
    --metrics \
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum \
    ./your_executable
```

**判断标准：**

```
bank_conflicts / wavefronts = 平均每 wavefront（warp）的 bank conflict 次数

< 1.05: 几乎没有 → 不需要 padding
1.05-1.5: 轻微 → 可以考虑 +4 padding
1.5-3.0: 中等 → 建议 padding
> 3.0: 严重 → 必须 padding！
```

---

## 3. 针对你的 GEMM V4 的完整分析脚本

```bash
#!/bin/bash
# profile_gemm_v4.sh

BIN=./build/bin/gemm_v4
KERNEL=GemmV4Kernel

echo "===== Step 1: Quick Scan ====="
ncu --kernel-name $KERNEL \
    --metrics \
    gpu__time_duration.sum,\
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    lauch__occupancy_per_register_file \
    $BIN

echo "===== Step 2: Stall Analysis ====="
ncu --kernel-name $KERNEL \
    --metrics \
    smsp__average_warps_issue_stalled_barrier_per_cycle_active.ratio,\
    smsp__average_warps_issue_stalled_long_scoreboard_per_cycle_active.ratio,\
    smsp__average_warps_issue_stalled_short_scoreboard_per_cycle_active.ratio,\
    smsp__average_warps_issue_stalled_not_selected_per_cycle_active.ratio,\
    smsp__average_warps_issue_stalled_math_pipe_throttle_per_cycle_active.ratio \
    $BIN

echo "===== Step 3: Memory Analysis ====="
ncu --kernel-name $KERNEL \
    --metrics \
    dram__bytes.sum,\
    dram__sectors_read.sum,\
    dram__sectors_write.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    lts__t_sectors_pipe_lsu_mem_global_op_ld.sum \
    $BIN

echo "===== Step 4: Bank Conflict ====="
ncu --kernel-name $KERNEL \
    --metrics \
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum \
    $BIN
```

---

## 4. Nsight Compute 报告解读优先级

不用从头到尾读完所有指标，按这个顺序看：

```
优先级 1（必看，5 秒扫一眼）：
  □ SM Throughput % — 这是最直接的"SM 有没有在干活"
  □ Memory Throughput % — 有没有吃满带宽
  □ Duration — 和上次对比

优先级 2（必看，30 秒）：
  □ Top 3 Stall Reasons — 线程最主要在等什么？
  □ Occupancy (%) — SM 上有多少活跃 warp
  □ Registers/Thread — 寄存器压力

优先级 3（有问题时看，2 分钟）：
  □ Bank Conflicts — 共享内存访问效率
  □ L1/L2 Hit Rate — cache 命中率
  □ DRAM sectors Read/Write — 实际访存量 vs 理论访存量

优先级 4（深度优化时看，5-10 分钟）：
  □ Memory workload analysis (核间均衡性)
  □ Warp state statistics (每个 stall reason 的详细分布)
  □ Scheduler statistics (warp issue 效率)
  □ Source code correlation (回到 CUDA C++ 对应行)
```

---

## 5. 常见性能问题与 Nsight Compute 信号

| 问题 | Nsight Compute 中的信号 | 解法 |
|------|------------------------|------|
| Occupancy 太低 | Theoretic Occupancy < 30% | 减少 regs/SMEM → 减小 tile size |
| 寄存器溢出 | `l1tex__t_sectors_pipe_lsu_mem_local_*` > 0 | 减少每线程的工作量 |
| Bank Conflict | `bank_conflicts/wavefronts > 2` | +4 padding |
| 访存不连续 | `dram__sectors_read` >> `dram__bytes_read / 32` | 用 float4；检查地址对齐 |
| 线程 load 不均 | 不同 thread 的 stall 分布差异大 | 重新设计 work partition |
| cp.async 没生效 | `cp.async` 指令数 = 0 | 检查编译器是否优化掉；用 asm volatile 强制 |
| Tensor Core 没被用 | SASS 中没有 `HMMA` 指令 | 检查数据类型和 API 用法 |

---

---
> 为 0，说明编译器没生成异步加载——这可能就是性能差的原因。
>
> 整个过程通常 5-10 分钟完成一个 kernel 的完整分析。分析完我会把结果
> 整理到 README 的 optimization notes 部分，形成版本间可追溯的优化记录。"

---

## 7. Nsight Compute 命令行速查

```bash
# ====== 常用命令 ======

# 列出可执行文件中的所有 kernel
ncu --list-kernels ./build/bin/gemm_v4

# 只分析特定 kernel
ncu --kernel-name GemmV4Kernel ./build/bin/gemm_v4

# 控制 launch 次数（减少 profiling overhead）
ncu --launch-count 1 --kernel-name GemmV4Kernel ./build/bin/gemm_v4

# 跳过前 N 次 launch（warmup 后再 profile）
ncu --launch-skip 5 --launch-count 3 --kernel-name GemmV4Kernel ./build/bin/gemm_v4

# 只收集特定 metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./build/bin/gemm_v4

# 导出为 CSV（方便在 Python/Pandas 中分析）
ncu --csv --kernel-name GemmV4Kernel ./build/bin/gemm_v4 > gemm_v4_metrics.csv

# 指定 GPU（多 GPU 系统中）
ncu --device 0 --kernel-name GemmV4Kernel ./build/bin/gemm_v4

# 生成 roofline 图（自动画图！）
ncu --set roofline --kernel-name GemmV4Kernel ./build/bin/gemm_v4

# 对比两个 kernel（版本间比较）
ncu --compare-blah-blah  # (需要 GUI)
```

---

## 8. 对你有用的 10 个核心指标

按常看频率排序：

```
1. gpu__time_duration.sum                          — kernel 耗时（和 benchmark 数据对比）
2. sm__throughput.avg.pct_of_peak_sustained_elapsed — SM 利用率（直接反映"有多忙"）
3. dram__throughput.avg.pct_of_peak_sustained_elapsed — DRAM 带宽利用率
4. lauch__occupancy_per_register_file              — Occupancy（活跃 warp 占比）
5. smsp__average_warps_issue_stalled_long_scoreboard_per_cycle_active.ratio — 等内存
6. smsp__average_warps_issue_stalled_barrier_per_cycle_active.ratio — 等同步
7. smsp__average_warps_issue_stalled_not_selected_per_cycle_active.ratio — 其他 warp 忙
8. l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum — Bank conflict (load)
9. sm__warps_active.avg.pct_of_peak_sustained_elapsed — 活跃 warp 占比
10. dram__bytes.sum                                — 总 DRAM 传输量（检查是否异常多）
```
