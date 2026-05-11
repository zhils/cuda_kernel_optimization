# CUDA Kernel 项目面试冲刺路线图

> 目标：在最短时间内，把当前项目从“强工程样例”升级到“面试可打高分的工业化证据链”。
> 评审口径：可证明（correctness/perf）、可复现（run_id + gate）、可解释（根因归因）、可落地（接口与迁移策略）。

## 0. 当前基线（你已经有）

- 多算子多版本优化（v0~vN），含 Nsight Compute 数据链路
- 分层 benchmark 套件（smoke/full/profile）
- run 级环境指纹、标准化结果、回归报告
- 性能门禁（baseline + WARN/FAIL）与自动调度初版

结论：已经超过多数“单 kernel 调优”项目，进入“准工业化”阶段。

---

## 1. 两周冲刺目标（面试导向）

### O1：模型级价值闭环（最高优先）

- 建一个 mini end-to-end 子图（建议：Q Path + Softmax + Output Gate）
- 输出：
  - 端到端 latency 对比（优化前/后）
  - 精度差异（max_abs_err/MAE）
  - 对应算子瓶颈归因摘要（不是只给总耗时）

### O2：跨架构可迁移策略（最小可证明）

- 给出 `sm120` 与另一代架构的策略差异说明（哪怕只 smoke）
- 输出：
  - dispatch 规则差异文档
  - 至少 1 组非 sm120 运行记录（或明确 fallback 行为证据）

### O3：回归归因自动化

- 当性能 gate 触发 WARN/FAIL 时，自动关联 NCU 关键指标
- 输出：
  - “退化原因初判”字段（memory-bound / math-pipe / occupancy / launch）
  - PR 可读摘要（Top N 异常）

### O4：面试叙事模板化

- 每个代表算子固定 1 页“问题-方法-证据-取舍-迁移”
- 输出：
  - `interview_packets/`（或单文档）标准模板
  - 3 个代表算子（GEMM / Softmax / QPath）完整样稿

---

## 2. 周计划（D1~D14）

### Week 1：证据链补齐

- D1-D2：确定 mini e2e 子图与统一输入规模，跑 baseline
- D3-D4：接入优化路径，形成 e2e 前后对比
- D5：补齐误差统计 + 运行稳定性统计（median/p90/std）
- D6：接 gate 与 report，形成“一次运行一份完整证据”
- D7：整理第一版面试材料（图 + 表 + 结论）

### Week 2：工业化与答辩能力

- D8-D9：跨架构或 fallback 证据补齐（至少 smoke）
- D10：回归归因自动化（WARN/FAIL -> 根因初判）
- D11-D12：3 个算子叙事模板定稿
- D13：高频追问演练（trade-off / 失败案例 / 下一步）
- D14：总彩排（15 分钟技术汇报 + 15 分钟追问）

---

## 3. 必交付产物清单（用于面试投递/现场展示）

- `INTERVIEW_SPRINT_ROADMAP.md`（本文件）
- e2e 对比报告（含吞吐、延迟、误差）
- 每个代表算子的证据页（至少 3 个）
- 一键复现命令（smoke/full/profile + gate）
- 一份“失败案例复盘”（最能体现工程成熟度）

---

## 4. 面试官高频问题与答法锚点

- 你如何证明优化真实有效？
  - 回答锚点：同输入同环境 + run_id + 统计分位数 + gate 阈值
- 退化时怎么排查？
  - 回答锚点：gate 告警 -> NCU 指标 -> stall top -> 定位改动
- 为什么这样选 kernel？
  - 回答锚点：dispatch 规则（arch/dtype/layout/shape_bucket）+ autotune 证据
- 换架构怎么办？
  - 回答锚点：路由规则 + fallback + 最小实测

---

## 5. 通过标准（可作为冲刺结束条件）

- 端到端子图：有明确前后提升，且误差在阈值内
- 任一核心算子：能在 3 分钟内讲清“瓶颈 -> 方案 -> 证据 -> trade-off”
- 新提交引入回退：门禁能自动拦截并给出初步归因
- 代码与文档：第三方按 README 可复现主要结论
