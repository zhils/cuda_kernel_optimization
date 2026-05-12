# Q Path Fusion

## 目标

将注意力模块中 Query 分支的两步计算（RMSNorm + Linear）融合到一条路径，减少中间张量全局内存读写。

---

## 数学定义

设输入 `X`（形状 `R x D`），`gamma`（形状 `D`）为 RMSNorm 权重，`W_q`（形状 `D x D`），`b_q`（形状 `D`）。

$$
N = \text{RMSNorm}(X; \gamma, \epsilon)
$$
纯文本：`N = RMSNorm(X, gamma, eps)`。

$$
Q = N \cdot W_q + b_q
$$
纯文本：`Q = N * W_q + b_q`。

---

## 版本规划

| 版本 | 做法 |
|:----|------|
| v0 | 朴素融合（正确性基线） |
| v1 | 并行归约 + SMEM 缓存 norm |
| v2 | Warp-per-row（block=128，每 lane 同时算 2 输出） |
| v2_bq_only | v2 对照版（只缓存 bq，wq 直接访存） |

---

## 构建

```bash
cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=120 && make q_path_fusion_v2 -j$(nproc)
cd ..
./build/bin/q_path_fusion_v2
```

---

## Nsight Compute 瓶颈分析（2026-05-09）

命令：`ncu --set basic --target-processes all --kernel-name-base demangled`。  
统计口径：每个版本取 Duration 最大的一次 launch。

| 版本 | 代表内核 | Max Duration(us) | Compute(SM) | DRAM | Memory | Achieved Occupancy | Reg/Thr | 结论 |
|:-----|:---------|-----------------:|------------:|-----:|-------:|-------------------:|--------:|:-----|
| `q_path_fusion_v0` | `QPathFusionV0Kernel` | 166.91 | 72.45% | 4.13% | 72.45% | 88.76% | 34 | 计算占主导，occupancy 较高 |
| `q_path_fusion_v1` | `QPathFusionV1Kernel` | 161.60 | 75.89% | 4.42% | 75.89% | 89.03% | 34 | 相比 v0 有小幅提升 |
| `q_path_fusion_v2` | `RMSNormKernel` | 697.89 | 17.81% | 81.28% | 81.28% | 95.80% | 18 | 以 RMSNorm 阶段的带宽瓶颈为主 |

原始报告：`data/ncu_reports/text/q_path_fusion_v0.txt`、`q_path_fusion_v1.txt`、`q_path_fusion_v2.txt`。

---

## Warp Stall 原因分析

| 版本 | #1 Stall | #2 Stall | #3 Stall | #4 Stall | #5 Stall |
|:----|:---------|:---------|:---------|:---------|:---------|
| v0 | **Long Scoreboard 79.7%** | Mio Throttle 7.9% | Wait 5.5% | Short Scoreboard 3.9% | Not Selected 2.7% |
| v1 | **Long Scoreboard 80.2%** | Mio Throttle 7.5% | Wait 5.5% | Short Scoreboard 3.9% | Not Selected 2.5% |
| v2 | Mio Throttle 27.2% | Short Scoreboard 26.8% | Long Scoreboard 15.4% | Not Selected 12.9% | No Instruction 8.5% |

v0/v1 以 Long Scoreboard 占主导（~80%），v2 的 stall 分布更均匀——Mio Throttle 27.2% + Short Scoreboard 26.8% + Long Scoreboard 15.4%。这与 v2 的瓶颈阶段变化一致：融合后的 RMSNorm 阶段成为带宽瓶颈，其 stall 模式从单纯等待全局内存过渡到 MIO 管道拥塞和缓存访问延迟的混合模式。

---

## 主场景性能口径（统一）

主指标统一为主场景 `gpu_ms`，NCU 吞吐仅用于瓶颈归因。

| 实现 | 主场景维度 | GPU耗时(ms) | 校验状态 | 数据文件 |
|---|---|---:|---|---|
| `q_path_fusion_v2` | `rows=1024,cols=1024` | 0.182544 | PASS | `data/results/q_path_fusion_v2_results.csv` |

环境口径：`RTX 5060 Ti (sm_120) + CUDA 13.2`。
统一汇总：`data/results/main_scenario_unified.csv`（retest tag: `20260512_manual_retest`）。

## 已知边界与后续补充

- 当前主口径使用 `rows=1024,cols=1024`（4096 档位为 `SKIP`），大尺寸正确性仍需单独补测。
- 建议补充不同 `cols` 桶下 `RMSNormKernel` 与 `QPathFusionV2Kernel` 的分阶段占比。
