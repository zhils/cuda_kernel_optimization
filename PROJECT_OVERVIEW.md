# CUDA Kernel Optimization 项目总览（细致版）

本文档用于系统梳理项目的目标、架构、算子矩阵、工程化体系、数据口径、运行方式与后续建设方向。  
建议配合 `README.md` 与 `docs/repro_and_regression_protocol.md` 一起阅读。

---

## 1. 项目定位

### 1.1 目标

本项目不是单纯的“写几个快 kernel”，而是构建一个可持续演进的高性能算子工程：

- **算子层**：覆盖常见深度学习核心算子与融合算子，按 v0->vN 迭代优化。
- **方法层**：形成“瓶颈判断 -> 单变量优化 -> 性能/正确性验证 -> 回归治理”的闭环。
- **工程层**：提供统一入口、统一口径、可追溯结果与性能门禁，支持长期维护。

### 1.2 项目特征

- 面向 CUDA C++ 的手写算子优化。
- 同时重视性能指标与数值正确性。
- 使用 Nsight Compute 进行瓶颈分析和 stall 归因。
- 具备 run 级归档、baseline 对比、回归 gate 的工程能力。

---

## 2. 核心能力总览

### 2.1 算子覆盖

项目当前覆盖以下算子家族：

- `gemm`：通用矩阵乘（FP32/FP16/INT8/参考库对比）
- `softmax`
- `rmsnorm`
- `flash_attention`
- `fused_conv1d_silu`
- `fused_gated_delta_rule`
- `fused_l2_norm_qk`
- `fused_output_norm_gate`
- `q_path_fusion`
- `pytorch_extension`（绑定示例）

### 2.2 工程能力

- 分层执行：`smoke` / `full` / `profile`
- 统一标准化结果（schema v1）
- 环境指纹（manifest）与 run_id 归档
- 性能回归门禁（WARN/FAIL 双阈值）
- 自动调优缓存（autotune cache）
- 统一调度（arch/dtype/layout/shape_bucket）
- NCU 结构化解析与 run 报告输出

---

## 3. 目录结构与职责

## 3.1 顶层目录（高价值部分）

- `common/`
  - 公共基础：测试样例、矩阵初始化、误差校验、CUDA 宏、配置结构
  - 代表文件：`benchmark.h`、`cuda_utils.h`、`kernel_config.h`
- `<op_name>/`
  - 各算子目录，包含 v0~vN 优化版本与本算子 README
- `scripts/`
  - 工程化脚本：回归、复现、归一化、报告、门禁、调度、autotune
- `configs/`
  - 调度配置中心：`kernel_catalog.json`
- `data/`
  - 结果、基线、归档报告（运行产生）
- `docs/`
  - 协议与专题文档（复现规范、量化数据等）
- `tests/`
  - 统一测试框架骨架（当前为可选构建项，尚在扩展）

## 3.2 关键脚本职责

- `scripts/benchmark_suite.sh`
  - 项目统一入口：`smoke/full/profile`
- `scripts/run_correctness_regression.sh`
  - 正确性回归、日志采集、失败判定、随机 case 入口
- `scripts/run_benchmark_repro.sh`
  - 可复现实验批跑 + 标准化汇总
- `scripts/normalize_results.py`
  - 异构 CSV 统一为 schema v1
- `scripts/check_performance_regression.py`
  - baseline 对比与回归 gate
- `scripts/update_performance_baseline.py`
  - 从标准化结果更新 golden baseline
- `scripts/update_autotune_cache.py`
  - 从标准化结果生成 family/bucket 级 kernel 偏好
- `scripts/kernel_dispatch.py`
  - 按 tier + 路由维度选择目标集合
- `scripts/generate_run_report.py`
  - 聚合生成 run 级 Markdown 报告
- `scripts/parse_ncu_reports.py`
  - 结构化解析 NCU 文本报告

---

## 4. 算子开发逻辑（统一方法论）

每个算子的优化遵循固定流程：

1. **基线版本（v0）**
   - 确保功能正确，形成可比较起点。
2. **瓶颈预判**
   - 使用算术强度与 roofline 判断偏 memory-bound 还是 compute-bound。
3. **单变量优化**
   - 每版尽量只改一个核心变量（tile、向量化、共享内存、warp reduce、融合边界等）。
4. **正确性校验**
   - 与 CPU/reference 对比，输出 PASS/FAIL 与误差指标。
5. **性能验证**
   - 记录 ms、吞吐、speedup；必要时用 NCU 分析 stall 根因。
6. **纳入工程链路**
   - 接入统一脚本、统一口径、统一回归门禁。

这保证“版本迭代可解释”，而不是黑盒调参。

---

## 5. 统一执行体系（P0 基线）

## 5.1 分层入口

```bash
bash scripts/benchmark_suite.sh smoke
bash scripts/benchmark_suite.sh full
bash scripts/benchmark_suite.sh profile
```

### `smoke`

- 快速正确性回归
- 适合本地改动后的第一轮检查

### `full`

- 正确性 + 可复现实验 + 标准化结果 + 报告
- 默认更新 autotune cache
- 默认执行性能回归门禁

### `profile`

- 独立 NCU 套件（all/stall/roofline）
- 结构化输出 `ncu_summary.csv`

## 5.2 run 级归档

每次执行会生成 `run_id`，产物归档至：

- `data/results/runs/<run_id>/...`
- `data/ncu_reports/<run_id>/...`

并落盘环境指纹：

- `data/results/runs/<run_id>/manifest.json`

---

## 6. 数据口径与结果文件

## 6.1 标准化结果（schema v1）

主文件：

- `data/results/summary_standardized.csv`
- `data/results/runs/<run_id>/summary_standardized.csv`

关键字段：

- 维度：`op_family`、`kernel_target`、`shape_kind`、`shape`
- 性能：`gpu_mean_ms`、`throughput_value`、`throughput_unit`、`speedup`
- 正确性：`verify_status`、`max_abs_err`、`threshold_used`
- 可追溯：`run_id`、`git_sha`、`gpu_name`、`driver_version`、`cuda_version`

## 6.2 正确性汇总

- `data/results/regression_summary.csv`
- `data/results/runs/<run_id>/regression_summary.csv`

字段包含：

- `status`
- `max_abs_err_worst`
- `pass_rows/fail_rows/skip_rows/not_run_rows`

## 6.3 run 报告

- `data/results/runs/<run_id>/report.md`

聚合回归、性能、门禁、NCU摘要，供快速审阅。

---

## 7. 性能回归门禁

## 7.1 baseline

生成或刷新：

```bash
python3 scripts/update_performance_baseline.py \
  --summary data/results/summary_standardized.csv \
  --output data/baselines/perf_golden.csv
```

## 7.2 gate 机制

`full` 默认启用：

- `PERF_GATE=1`
- WARN/FAIL 双阈值（ms 回退、吞吐下降）
- FAIL 返回非零，便于 CI 直接拦截

产物：

- `performance_regression_check.csv`
- `performance_gate_summary.md`（PR 友好）

---

## 8. 调度与自动调优

## 8.1 调度配置中心

- `configs/kernel_catalog.json`
  - 定义各 tier 的默认目标列表
  - 定义 family 归属
  - 定义 `routing_rules`（按 arch/dtype/layout/shape_bucket 覆写）

## 8.2 调度执行器

- `scripts/kernel_dispatch.py`
  - 输入：tier + 路由维度 + autotune cache
  - 输出：去重后的目标列表

路由维度：

- `DISPATCH_ARCH`
- `DISPATCH_DTYPE`
- `DISPATCH_LAYOUT`
- `DISPATCH_SHAPE_BUCKET`

## 8.3 autotune cache

- `data/baselines/autotune_cache.json`
  - `preferred_by_family`
  - `preferred_by_family_by_bucket`

更新来源：`summary_standardized.csv`（PASS 且有吞吐数据）

---

## 9. 正确性策略与随机鲁棒性

默认策略：

- 小规模与参考实现对比，输出 PASS/FAIL
- 大规模超出校验预算时标记 `NOT_RUN`（避免误导性 PASS）

随机鲁棒性入口（当前已接入）：

```bash
RANDOM_GEMM_CASES=1 \
RANDOM_SOFTMAX_CASES=1 \
RANDOM_QPATH_CASES=1 \
bash scripts/run_correctness_regression.sh
```

机制：

- 自动备份原始 `test_cases.csv`
- 覆盖为随机 case 执行
- 退出后自动恢复原文件

---

## 10. NCU 分析体系

入口脚本：

- `run_ncu_all.sh`
- `run_ncu_stall.sh`
- `run_ncu_roofline.sh`

特点：

- 与功能 benchmark 解耦
- run_id 归档
- 支持 quick 模式
- 默认 fail-fast，可 `ALLOW_NCU_FAIL=1` 降级告警

结构化输出：

- `data/ncu_reports/<run_id>/ncu_summary.csv`

---

## 11. `tests/` 现状说明（重要）

项目当前已建立统一脚本级入口（`benchmark_suite.sh` + 回归/复现脚本）并在全项目生效。  
`tests/` 目录中的统一测试框架（`unified_test`）处于骨架阶段，尚未完成全部算子迁移注册。

建议认知：

- **现在可用**：脚本级统一验证体系（已用于主流程）
- **后续建设**：将算子逐步迁移到 `tests/` 注册式统一入口

---

## 12. 常用命令速查

### 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j
```

### 全链路

```bash
bash scripts/benchmark_suite.sh smoke
bash scripts/benchmark_suite.sh full
bash scripts/benchmark_suite.sh profile
```

### 只跑单算子

```bash
cmake --build build --target gemm_v2
./build/bin/gemm_v2
```

### 自定义调度维度示例

```bash
DISPATCH_ARCH=sm120 \
DISPATCH_DTYPE=fp32 \
DISPATCH_LAYOUT=row_major \
DISPATCH_SHAPE_BUCKET=large \
bash scripts/run_benchmark_repro.sh
```

---

## 13. 已知边界与后续建议

### 已知边界

- 统一 `tests/` 注册入口未覆盖全部算子
- 跨架构验证数据仍以单架构为主
- 端到端模型级收益链路（mini graph）仍可加强

### 后续建议（优先级）

1. 完成 `tests/` 统一入口迁移（先 3 个代表算子）
2. 增加 e2e 子图性能/精度闭环报告
3. 引入 gate 触发后的自动根因初判（关联 NCU 指标）
4. 扩展跨架构 smoke 路由与 fallback 证据

---

## 14. 参考文档

- 项目主页：`README.md`
- 协议文档：`docs/repro_and_regression_protocol.md`
- 面试冲刺：`INTERVIEW_SPRINT_ROADMAP.md`
- 面试口播：`INTERVIEW_PRESENTATION_SCRIPT.md`
