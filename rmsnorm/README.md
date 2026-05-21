# RMSNorm CUDA 优化复盘

---

## 1. 数学定义

$$
y = \frac{x}{\sqrt{\mathrm{mean}(x^2)+\epsilon}}\cdot \gamma
$$

纯文本：`y = x / sqrt(mean(x^2) + eps) * gamma`。

$$
\mathrm{mean}(x^2)=\frac{1}{C}\sum_{i=0}^{C-1}x_i^2
$$

纯文本：`mean(x^2) = (1/C) * sum_{i=0..C-1}(x_i^2)`。

其中 `x` 形状为 `(R,C)`，`gamma` 形状为 `(C)`，`eps = 1e-5`（数值稳定常数）。

**算术强度：**
- 计算量 ≈ 4×R×C FLOPs（平方和 + rsqrt + 乘法缩放 + gamma 乘）
- 最小搬运 = R×C×4×2 bytes（读 x + 写 y，gamma 是 C 大小，可忽略）
- 强度 ≈ 4RC / 8RC = 0.5 FLOP/Byte → **典型 memory-bound**

RMSNorm 每个元素只做 O(1) 次计算，带宽决定速度。

---

## 2. 版本演进

| 版本 | 改造点 | 4096² 耗时 | 带宽 (GB/s) |
|:----|--------|:----------:|:-----------:|
| v0 | 每行单线程串行 | 1.4220 ms | 94 |
| v1 | SMEM staging + float4 | 0.6816 ms | 197 |
| v2 | + warp shuffle 归约 | 0.4547 ms | 295 |
| v3 | weight 缓存到 SMEM | **0.3719 ms** | **361** |

实测日期：**2026-05-20**。

v1/v2/v3 在 4096² 差距不大（~0.35–0.38 ms），因为大尺寸下带宽已经触顶。小尺寸（128~1024）v3 优势明显。

### v0 — 每行单线程（基线）

```cuda
// 每行一个线程：串行算平方和 → rsqrt → 写回
for (int row = blockIdx.x; row < rows; row++) {
    float sum = 0;
    // 读一遍 x 算平方和
    for (int j = 0; j < cols; j++)
        sum += x[row * cols + j] * x[row * cols + j];
    float rms = rsqrt(sum / cols + eps);
    // 再读一遍 x 做归一化
    for (int j = 0; j < cols; j++)
        y[row * cols + j] = x[row * cols + j] * rms * gamma[j];
}
```
问题：x 读两遍，gamma 每行从全局内存读一遍，没有 block 内协作。**4096²: 1.4220 ms / 94 GB/s。**

### v1 — SMEM staging + float4 向量化

每行交给一个 block，SMEM 里做平方和归约，`float4` + `__ldg` 一次性读 4 个 float。

```
线程映射：grid(rows) × block(cols/4)
每个线程：从全局用 float4 读 x，存 SMEM
          归约 x² → rms
          从 SMEM 读回 x，做归一化，float4 写 y
```
gamma 还是每行从全局读——**这是 v3 要解决的事。**
**4096²: 0.6816 ms / 197 GB/s。**

### v2 — Warp shuffle 归约

v1 的 SMEM 归约（tree reduce）换成了 `__shfl_xor_sync` 寄存器归约：

```cuda
float sum = x_part * x_part;
// warp shuffle 树归约
sum += __shfl_xor_sync(0xffffffff, sum, 16);
sum += __shfl_xor_sync(0xffffffff, sum, 8);
sum += __shfl_xor_sync(0xffffffff, sum, 4);
sum += __shfl_xor_sync(0xffffffff, sum, 2);
sum += __shfl_xor_sync(0xffffffff, sum, 1);
// lane 0 得到完整的平方和
```

少了一次 SMEM 写回+读出的 round trip。**4096²: 0.4547 ms / 295 GB/s。**

### v3 — Weight 缓存到 SMEM + Warp 归约

gamma 一开始就从全局加载到动态 SMEM 并广播给所有 warp。

```
线程映射：block(128) = 4 warp，每个 warp 处理一行
SMEM 排布：
  gamma_smem[cols]  ← 启动时一次性加载 (cudaMemcpyToSymbol 风格)
                       每个线程协作搬 cols/128 个元素

归约路径：
  x² → warp shuffle sum → 存 SMEM → cross-warp SMEM 归约 → rms

读写路径：
  对齐时 float4，不对齐时 float
```

**4096²: 0.3719 ms / 361 GB/s。** 占理论带宽（448 GB/s）的 **81%**。

### v3 多 dtype 扩展（P0 已落地）

**入口：** 单 binary `rmsnorm_v3`，CLI 选择 dtype（不新增 v4 演进线）。

```bash
./build/bin/rmsnorm_v3                      # 默认 fp32（与历史 benchmark 一致）
./build/bin/rmsnorm_v3 --dtype fp16
./build/bin/rmsnorm_v3 --dtype bf16
./build/bin/rmsnorm_v3 --dtype fp32 --weight-dtype fp32   # 显式指定 weight
```

**设计决策（已确认）：**

| 议题 | 决策 |
|------|------|
| IO 是否同 dtype | **不强制**；P0 中 x/y 跟随 `--dtype`，计算在 FP32 累加 |
| weight dtype | **始终 fp32/fp16/bf16**（不支持 FP8/INT8 weight） |
| 量化对象 | **仅 activation**；weight 保持高精度 |
| FP8/INT8 scale | **动态 per-row**（Phase 2，非静态 tensor scale） |
| FP8 变体 | **E4M3**（激活常用）+ **E5M2**（梯度常用），CLI 已预留 |
| P0 范围 | **fp32 / fp16 / bf16** |
| 演进关系 | **只在 v3 上扩展**，v0–v2 保持 FP32 历史对比 |

**P0 实测（4096²，2026-05-20 复跑）：**

| `--dtype` | GPU ms | 带宽 (GB/s) | 校验 |
|-----------|-------:|------------:|:----:|
| fp32 | 0.372 | 361 | PASS |
| fp16 | 0.177 | 379 | PASS |
| bf16 | 0.171 | 392 | PASS |

> fp16/bf16 带宽按 `sizeof(dtype)×2` 统计（读 x + 写 y）；算子在 SMEM 中仍以 **float 累加** weight 与平方和。

**Phase 2 已实现（动态 per-row scale）：**

```bash
./build/bin/rmsnorm_v3 --dtype int8              # weight 默认 fp32
./build/bin/rmsnorm_v3 --dtype fp8_e4m3          # 激活常用 E4M3
./build/bin/rmsnorm_v3 --dtype fp8_e5m2          # 梯度常用 E5M2
./build/bin/rmsnorm_v3 --dtype int8 --weight-dtype fp16
```

| 约定 | 说明 |
|------|------|
| 输入 | `x_q[rows,cols]` + `x_scale[rows]`（`scale = max_abs/quant_max`） |
| 输出 | `y_q[rows,cols]` + **动态** `y_scale[rows]`（kernel 写回） |
| 计算 | dequant → FP32 RMSNorm → requant；weight 始终高精度 |
| IO 同 dtype | **不强制**；校验在 dequant 后的 FP32 域比较 |

**Phase 2 实测（4096²，weight=fp32）：**

| `--dtype` | GPU ms | 等效带宽* | 校验 |
|-----------|-------:|----------:|:----:|
| int8 | 0.167 | 201 GB/s | PASS |
| fp8_e4m3 | 0.152 | 221 GB/s | PASS |
| fp8_e5m2 | 0.152 | 220 GB/s | PASS |

\* 量化 dtype 带宽按 `1B×2`（读 x + 写 y）统计，不含 scale 向量。

实现文件：`rmsnorm/rmsnorm_quant.h`（host quant/CPU ref）、`rmsnorm/rmsnorm_v3_dtype.cuh`（INT8/FP8 kernel）。

---

## 3. 性能数据

实测日期：**2026-05-20**

### 3.1 执行时间（ms）

| Rows | Cols | V0 | V1 | V2 | V3 | CUB |
|------|------|-----|-----|-----|-----|-----|
| 128 | 128 | 0.0221 | 0.0053 | 0.0045 | **0.0048** | — |
| 256 | 256 | 0.0822 | 0.0069 | 0.0079 | **0.0044** | — |
| 512 | 512 | 0.1618 | 0.0104 | 0.0066 | **0.0084** | — |
| 1024 | 1024 | 0.3350 | 0.0289 | 0.0092 | **0.0108** | — |
| 4096 | 4096 | 1.4220 | 0.6816 | 0.4547 | **0.3719** | 0.3518* |

`*` CUB ref 4096² 为 2026-05-20 早期复测值。

### 3.2 带宽（GB/s）

| Rows | Cols | V0 | V1 | V2 | V3 |
|------|------|------|------|------|------|
| 128 | 128 | 5.9 | 24.7 | 29.3 | **27.2** |
| 256 | 256 | 6.4 | 76.2 | 66.3 | **119.9** |
| 512 | 512 | 13.0 | 201.7 | 319.5 | **250.7** |
| 1024 | 1024 | 25.0 | 290.1 | 909.3 ⚡ | **775.1** ⚡ |
| 4096 | 4096 | 94.4 | 196.9 | 295.2 | **360.9** |

> 1024² / 512² 带宽 > 448 GB/s：数据在 L2 cache 内，非纯 DRAM。

---

## 4. Nsight Compute 瓶颈分析

> **口径：** stall 来自 `ncu -c 1` 第一次 launch；basic 指标沿用 2026-05-09 大 launch 数据。

命令：`ncu --set basic --target-processes all --kernel-name-base demangled`。  
统计口径：每个版本取 Duration 最大的一次 launch。

| 版本 | Max Duration(us) | Compute(SM) | DRAM | Memory | Achieved Occupancy | Reg/Thr | 结论 |
|:-----|-----------------:|------------:|-----:|-------:|-------------------:|--------:|:-----|
| `rmsnorm_v0` | 908.22 | 0.95% | 4.55% | 12.81% | 16.48% | 39 | 基线版本利用率很低 |
| `rmsnorm_v1` | 697.54 | 51.31% | 39.07% | 51.31% | 8.33% | 40 | 算存均衡，但 occupancy 受共享内存限制 |
| `rmsnorm_v2` | 321.18 | 10.20% | 82.86% | 82.86% | 8.33% | 40 | 转为明显带宽受限 |
| `rmsnorm_v3` | 334.18 | 5.88% | 86.90% | 86.90% | 38.68% | 40 | DRAM 接近饱和，符合 memory-bound 预期 |

结论：v2/v3 的优化方向本质是把瓶颈“推”到内存带宽上；v3 的吞吐上限主要由 DRAM 决定。

---

## Warp Stall 原因分析（2026-05-20 全量 NCU）

统计口径：每个 binary 取**第一次 kernel launch** 的 7 项 stall 指标（`build/data/ncu_reports/retest_20260520T033426Z/stall/`）。

| 版本 | Long SB | Short SB | Wait | MIO Thr | Math Pipe | 总 Stall | 结论 |
|:----|--------:|---------:|-----:|--------:|----------:|---------:|:-----|
| v0 | **80.07** | 0.26 | 2.52 | 0.00 | 0.00 | 83.20 | 等全局内存，利用率极低 |
| v1 | 3.83 | 1.31 | 1.94 | 0.39 | 0.00 | 8.62 | SMEM 协作 + Wait 占比高 |
| v2 | **9.61** | 2.45 | 2.17 | 0.00 | 0.02 | 16.76 | Long SB 主导，带宽受限 |
| v3 | **11.55** | 2.70 | 2.19 | 0.00 | 0.02 | 19.75 | DRAM 接近饱和 |

---

## 5. PTX/SASS

PTX 和 SASS 可在本地通过 `cuobjdump -ptx <binary>` 或 `cuobjdump -sass <binary>` 生成（`**/asm/` 已从版本控制中排除）。

关键 PTX 指令：
- 向量化加载：`ld.global.nc.v4.f32`
- warp 归约：`shfl.sync.down.b32`
- 快速倒数平方根：`rsqrt.approx.ftz.f32`

---

## 6. 产物路径

- 可执行文件：`build/bin/rmsnorm_v0` … `rmsnorm_v3`（v3 支持 `--dtype`）
- dtype 模块：`rmsnorm/rmsnorm_dtype.h`、`rmsnorm/rmsnorm_quant.h`、`rmsnorm/rmsnorm_v3_dtype.cuh`
- 结果 CSV：`data/results/rmsnorm_v3_results.csv`（含 `act_dtype,weight_dtype` 列）
- ncu 报告：`data/ncu_reports/`
- PTX/SASS：本地运行 `cuobjdump -ptx <binary>` 生成

---

## 主场景性能口径（统一）

实测日期：**2026-05-20**

| 实现 | 主场景维度 | GPU耗时(ms) | 带宽(GB/s) | 校验状态 |
|---|---|---:|---:|---|
| `rmsnorm_v3` (`--dtype fp32`) | `rows=4096,cols=4096` | **0.372** | 361 | PASS |
| `rmsnorm_v3` (`--dtype fp16`) | `rows=4096,cols=4096` | **0.177** | 379 | PASS |
| `rmsnorm_v3` (`--dtype bf16`) | `rows=4096,cols=4096` | **0.171** | 392 | PASS |
| `rmsnorm_v3` (`--dtype int8`) | `rows=4096,cols=4096` | **0.167** | 201 | PASS |
| `rmsnorm_v3` (`--dtype fp8_e4m3`) | `rows=4096,cols=4096` | **0.152** | 221 | PASS |
| `rmsnorm_v3` (`--dtype fp8_e5m2`) | `rows=4096,cols=4096` | **0.152** | 220 | PASS |
| `rmsnorm_cub_ref` | `rows=4096,cols=4096` | 0.352 | 382 | SKIP |

环境口径：`RTX 5060 Ti (sm_120) + CUDA 13.2`。
