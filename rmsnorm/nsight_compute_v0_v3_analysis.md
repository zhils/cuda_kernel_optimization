# RMSNorm v0~v3 Nsight Compute 分析记录

## 1. 分析目标

使用 Nsight Compute 从 SM 内部视角分析 `rmsnorm_v0_naive`、`rmsnorm_v1`、`rmsnorm_v2`、`rmsnorm_v3` 的实现问题：

1. 哪些版本受内存等待影响更严重？
2. 哪些版本受调度/并发不足影响更明显？
3. 为什么 v2/v3 在不同尺寸下会出现性能交叉？

---

## 2. 采集方式

## 2.1 采集对象

- `rmsnorm_v0_naive.exe`
- `rmsnorm_v1.exe`
- `rmsnorm_v2.exe`
- `rmsnorm_v3.exe`

## 2.2 采样参数

- 工具：`Nsight Compute 2025.2.0`
- 采样策略：`--launch-skip 3 --launch-count 1`（统一抓取代表性 launch）
- 指标：
  - `sm__throughput.avg.pct_of_peak_sustained_elapsed`
  - `dram__throughput.avg.pct_of_peak_sustained_elapsed`
  - `sm__warps_active.avg.pct_of_peak_sustained_active`
  - `smsp__average_warps_issue_stalled_*`

## 2.3 输出文件

- 汇总：`build/artifacts/ncu/rmsnorm/rmsnorm_v0_v3_ncu_summary.csv`
- 本文档：`rmsnorm/nsight_compute_v0_v3_analysis.md`

---

## 3. 核心结果汇总

| 版本 | SM 吞吐 %Peak | DRAM 吞吐 %Peak | Active Warps %Peak | stall_long_sb | stall_short_sb | stall_mio | stall_lg | stall_not_selected |
|------|---------------|------------------|--------------------|---------------|----------------|-----------|----------|--------------------|
| v0 | 0.78 | 2.45 | 16.60 | 110.70 | 0.03 | 0.00 | 0.21 | 0.01 |
| v1 | 18.85 | 6.83 | 48.13 | 1.50 | 4.89 | 2.85 | 0.00 | 0.71 |
| v2 | 10.47 | 5.89 | 52.58 | 1.75 | 3.21 | 0.71 | 0.00 | 0.72 |
| v3 | 3.66 | 13.93 | 12.22 | 5.11 | 3.22 | 0.01 | 0.00 | 0.07 |

---

## 4. 分版本问题分析

## 4.1 v0（Naive）

主要问题：

- `stall_long_sb` 极高（110.70），典型“等待长延迟数据返回”；
- `SM 吞吐` 与 `DRAM 吞吐` 都很低，说明整体执行效率差；
- `Active Warps` 也偏低，难以通过并发掩蔽延迟。

结论：

- v0 的核心问题是“串行访存 + 长依赖等待”，和 Naive 实现特征一致。

## 4.2 v1（Staged）

主要问题与改进：

- `stall_long_sb` 从 v0 的 110.70 显著降到 1.50，说明内存等待已大幅缓解；
- `Active Warps` 提升到 48.13%，并发明显改善；
- 但 `stall_short_sb` 与 `stall_mio` 仍有一定占比，表示流水仍可继续优化。

结论：

- v1 成功解决了 v0 的主要瓶颈，但进入“细粒度调度/流水”阶段。

## 4.3 v2（Vectorized + Warp 归约）

主要问题与改进：

- `Active Warps` 在四版中最高（52.58%），并发组织最优；
- `stall_mio`（0.71）与 `stall_short_sb`（3.21）相对 v1 更低，说明数据路径更顺；
- 但 `SM 吞吐`并未超过 v1，反映出“并发更高但计算单元利用未同步提升”。

结论：

- v2 在“并发组织和等待控制”上更稳健，是当前最均衡的实现之一。

## 4.4 v3（Fused）

主要问题：

- `DRAM 吞吐`最高（13.93%），但 `SM 吞吐`很低（3.66%）；
- `Active Warps` 仅 12.22%，并发不足；
- `stall_long_sb` 回升到 5.11，说明融合路径下出现新的长依赖链条。

结论：

- v3 的问题是“内存流量上去了，但计算侧没吃满”，融合策略当前实现不够平衡。

---

## 5. 跨版本结论

1. **v0 的本质问题已明确**：严重 long scoreboard，属于典型慢访存+低并发基线。  
2. **v1/v2 是有效演进**：两者都显著抑制了 long scoreboard，并提升并发。  
3. **v2 更稳健**：在 active warps 与 stall 控制上优于 v1。  
4. **v3 存在失衡**：DRAM 利用高但 SM 利用和并发低，导致整体收益不稳定。  

---

## 6. 下一步优化建议

1. **优先保留 v2 为主路径**  
   v2 在并发与 stall 指标上更均衡，可作为默认高性能实现。

2. **针对 v3 做并发修复**
   - 提升活跃 warps（检查 block/线程配置与寄存器压力）；
   - 缩短关键依赖链，降低 `stall_long_sb`；
   - 验证融合阶段是否引入了额外同步/串行段。

3. **联合 Nsight Systems 交叉验证**
   - NCU 解释“为什么慢”（SM 内部）；
   - NSYS 解释“时间花在哪”（系统级）；
   - 两者结合可以避免误判。

---

## 7. 待补截图标记（你后续补）

- [TODO-NCU-RMS-1] v0 的 Stall Reasons（突出 long scoreboard）
- [TODO-NCU-RMS-2] v1/v2 的 Active Warps 与 Stall 对比图
- [TODO-NCU-RMS-3] v3 的 DRAM 吞吐高但 SM 吞吐低的对照图
- [TODO-NCU-RMS-4] v2 vs v3 的 Source Correlation 片段截图

