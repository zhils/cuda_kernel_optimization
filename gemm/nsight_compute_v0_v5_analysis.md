# GEMM v0~v5 Nsight Compute 分析记录

## 1. 分析目标

使用 Nsight Compute 从 **SM 内部执行行为**（吞吐、活跃 warps、stall 原因）角度分析 `v0~v5`：

1. 各版本当前实现的主要问题是什么？
2. 为什么有些版本在 benchmark 中提速明显，有些版本会退化？
3. 下一步应优先优化哪里？

---

## 2. 采集方法

## 2.1 被测程序

- `gemm_v0_naive.exe`
- `gemm_v1.exe`
- `gemm_v2.exe`
- `gemm_v3.exe`
- `gemm_v4.exe`
- `gemm_v5.exe`

## 2.2 采样策略

- 使用同一组参数做横向对比：
  - `--launch-skip 33 --launch-count 1`
- 目的：抓取同一位置的代表性 launch（对应 1024 尺寸段）用于版本对比。

> 注：每个可执行程序内部会跑多组尺寸；本报告是“代表性 launch”分析，不是全部 launch 的平均值。

## 2.3 采集指标

- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__warps_active.avg.pct_of_peak_sustained_active`
- `smsp__average_warps_issue_stalled_*`（long scoreboard / lg throttle / mio throttle / not selected / short scoreboard）

## 2.4 输出文件

- 汇总 CSV：`build/artifacts/ncu/gemm/gemm_v0_v5_ncu_summary.csv`
- 本文档：`gemm/nsight_compute_v0_v5_analysis.md`

---

## 3. 核心结果汇总

| 版本 | SM 吞吐 %Peak | DRAM 吞吐 %Peak | Active Warps %Peak | stall_lg | stall_long_sb | stall_mio | stall_not_selected |
|------|---------------|------------------|--------------------|----------|---------------|-----------|--------------------|
| v0 | 95.30 | 1.92 | 97.78 | 29.26 | 5.77 | 0.07 | 4.80 |
| v1 | 94.72 | 3.32 | 97.77 | 0.00 | 3.58 | 34.55 | 5.13 |
| v2 | 32.35 | 3.79 | 22.23 | 0.15 | 2.01 | 0.74 | 0.87 |
| v3 | 43.86 | 7.21 | 27.17 | 0.20 | 2.90 | 0.67 | 0.84 |
| v4 | 41.40 | 9.44 | 28.09 | 0.53 | 1.09 | 0.56 | 1.33 |
| v5 | 30.38 | 5.15 | 30.58 | 1.23 | 27.47 | 1.77 | 0.14 |

---

## 4. 分版本问题分析

## 4.1 v0（Naive）

主要问题：

- `stall_lg` 很高（29.26），表明负载/存储路径受限明显；
- `stall_long_sb` 也不低，说明等待长延迟依赖（典型全局访存返回）；
- 虽然 `SM throughput` 与 `active warps` 看起来高，但有效计算效率并不高，说明“忙但不高效”。

结论：

- v0 的核心不是“并行度不足”，而是“访存路径堵塞 + 访存等待”。

## 4.2 v1（共享内存分块）

主要问题：

- `stall_mio` 异常高（34.55），说明内存输入输出管线（MIO）出现新的节流点；
- 相比 v0，瓶颈从 `lg_throttle` 转移到了 `mio_throttle`，属于“换了瓶颈但没根治”。

结论：

- v1 方案减少了部分全局访存压力，但共享内存/数据通路组织本身带来新的节流。

## 4.3 v2（线程级寄存器分块）

主要问题与改进：

- `stall_*` 指标整体明显下降，说明核心等待问题缓解；
- 但 `active warps` 下降到 22.23%，存在并发度偏低/寄存器压力带来的占用下降风险。

结论：

- v2 的问题从“访存堵塞”转为“占用率与并发度”问题，属于典型优化阶段性转移。

## 4.4 v3（寄存器预取）

主要问题：

- 相比 v2，`SM throughput` 上升，但 `stall_long_sb` 回升（2.90）；
- 说明预取策略没有完全掩蔽延迟，仍有等待链条。

结论：

- v3 并非对 v2 的绝对改进，收益依赖参数与尺寸；需要条件化启用。

## 4.5 v4（更大 tile）

主要问题与改进：

- `DRAM throughput` 在 v0~v5 中最高（9.44），数据通路利用更充分；
- `stall_long_sb` 降低到 1.09，说明等待全局返回的压力显著缓解。

结论：

- v4 在“访存效率和延迟隐藏”上是当前最均衡的一版之一，特别适合大尺寸。

## 4.6 v5（WMMA FP16->FP32）

主要问题：

- `stall_long_sb` 很高（27.47），这是当前最突出的风险信号；
- 说明即使引入 WMMA，仍可能因为数据供给或调度链路导致长延迟等待。

结论：

- v5 不是“上 Tensor Core 就自动最优”，当前实现主要瓶颈变成了 **Tensor Core 供数/调度链路**，需要继续打磨流水与数据搬运。

---

## 5. 跨版本共性结论

1. 瓶颈在版本间会迁移：  
   `v0/v1` 偏访存节流 -> `v2/v3` 偏占用与调度 -> `v5` 偏长依赖等待。

2. “吞吐高”不等于“效率高”：  
   需要同时看 stall 来源和活跃 warps，而不是只看单一吞吐指标。

3. v4 是当前较稳健方案，v5 是高潜力但未完全收敛方案：  
   v5 后续优化空间大，但当前还有明显长延迟链条问题。

---

## 6. 下一步优化建议（按优先级）

1. **优先解决 v5 的 long_scoreboard**
   - 强化 K 维流水深度；
   - 检查 shared/register 双缓冲是否真正重叠；
   - 减少关键路径上的依赖串行。

2. **做分尺寸调度策略**
   - `v4` 作为稳健默认；
   - `v5` 仅在特定尺寸和参数区间启用。

3. **补充 Nsight Systems 联合验证**
   - NCU 解释 SM 内部原因；
   - NSYS 解释系统级时间分布；
   - 两者结合避免误判。

---

## 7. 待补截图标记（你后续补）

- [TODO-NCU-GEMM-1] v0 的 Warp Stall Reasons 截图（突出 LG throttle）
- [TODO-NCU-GEMM-2] v1 的 MIO throttle 截图（说明瓶颈迁移）
- [TODO-NCU-GEMM-3] v2/v3/v4 的 Occupancy + Stall 对比截图
- [TODO-NCU-GEMM-4] v5 的 Long Scoreboard 截图（解释 WMMA 供数问题）
- [TODO-NCU-GEMM-5] v4 vs v5 的 Memory Workload Analysis 对比图

