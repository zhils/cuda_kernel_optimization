# CUDA Kernel 优化

---

## 三算子架构

| 类型 | 算子 | 代表问题 | 优化方向 | 主指标 |
|------|------|----------|----------|--------|
| **计算密集** | [GEMM](gemm/) | 算术强度 N/6≈683，Roofline 计算受限 | SMEM 分块 → cp.async → Tensor Core | TFLOPS |
| **访存密集** | [RMSNorm](rmsnorm/) | 算术强度 ≈0.5，远低于 Ridge Point | float4 向量化 + warp shuffle 归约 | GB/s |
| **融合算子** | [Fused Conv1D+SiLU](fused_conv1d_silu/) | 多 kernel 中间缓冲 + launch 开销 | kernel fusion，消除中间 tensor | 端到端 ms |

---

## 性能摘要（RTX 5060 Ti, sm_120, CUDA 13.2）

实测日期：**2026-05-20**

| 算子 | 主场景 | GPU 耗时 | 吞吐 | 实际利用率 | 校验 |
|------|--------|---------|------|-----------|------|
| GEMM FP32 (`gemm_v3`) | 4096³ | 11.50 ms | 11.95 TFLOPS | 51%（vs 23.5T CUDA Core） | PASS ≤1024 |
| GEMM FP16 (`gemm_fp16`) | 4096³ | 3.86 ms | 35.58 TFLOPS | 38%（vs 94T TC 实际上限） | cos_sim=1.0 |
| cuBLAS FP32 | 4096³ | 9.00 ms | 15.28 TFLOPS | 65%（vs 23.5T CUDA Core） | — |
| cuBLAS FP16 | 4096³ | 3.12 ms | 44.04 TFLOPS | 47%（vs 94T TC 实际上限） | — |
| cuBLAS FP8e4 | 4096³ | 0.84 ms | 163.78 TFLOPS | 41%（vs 0.4 POPS TC 实际上限） | — |
| RMSNorm (`rmsnorm_v3`) | 4096² | 0.372 ms | 361 GB/s | 81%（vs 448 GB/s DRAM） | PASS |
| Fused Conv1D+SiLU (`v3`) | B=8,L=2048 | 1.65 ms | ~343× vs v0 | — | PASS |

---

## 输入数据兼容性

所有算子 API 支持多种数据类型，cover 从开发训练到推理部署的完整链路：

| 数据类型 | GEMM | RMSNorm | Fused Conv1D+SiLU | 典型用途 |
|---------|------|---------|-------------------|---------|
| **FP32** | ✅ V3 hand-written + cuBLAS | ✅ V0/V1/V2/V3/kCubRef | ✅ V0/V1/V2/V3 | 训练基线、正确性参考 |
| **FP16** | ✅ WMMA + cuBLAS fp16 | ✅ | ✅ | 推理/Tensor Core 加速 |
| **BF16** | ✅ cuBLAS bf16 | ✅ | ✅ | 训练（不溢出） |
| **INT8** | ✅ cuBLAS int8 | ✅ | ✅ | 量化推理 |
| **FP8 E4M3** | ✅ cuBLAS fp8 | ✅ | ✅ | H100+ 稀疏推理 |
| **FP8 E5M2** | ✅ cuBLAS fp8 | ✅ | ✅ | H100+ 动态范围推理 |

**API 统一接口：**
- GEMM: `GemmRun(GemmParams)` 通过 `dtype_a/b/c` 指定精度，`impl` 选择实现路径
- GEMM 量化: `GemmQuantizedRun()` 支持 per-tensor / per-row 量化方案
- RMSNorm: `RmsNormRun(RmsNormParams)` 通过 `act_dtype/weight_dtype` 指定精度
- Fused Conv1D: `FusedRun(FusedParams)` 通过 `dtype` 统一指定输入/输出精度

**对齐与回退策略：**
- 手写 kernel（GEMM V3/FP16、RMSNorm V0-V3、Fused V0-V3）要求 tile 对齐（默认 128）
- 非对齐输入自动回退到 cuBLAS（GEMM）或 fp32 兼容路径（Fused Conv1D）
- `AlignmentPolicy` 控制行为：`kFallback`（回退）/ `kStrict`（报错）/ `kSkip`（跳过）

---

## 瓶颈分析

```
v1 SMEM 分块  → MIO Throttle 4.82   访存管道拥塞（128³）
v2 寄存器分块 → Long SB 低           cp.async 前全局延迟已缓解
v3 cp.async   → Long SB 0.37         FP32 最优 11.95 TFLOPS @ 4096³（51% CUDA Core 峰值）
v4 TF32 WMMA  → Math Pipe 8.79       TC 工作（128³ launch），TC 实际利用 ~42%（vs 24T 上限）
fp16 WMMA     → Long SB 1.77         TC 实际利用 ~38%（vs 94T 上限），DRAM ~2%（非带宽瓶颈）
cuBLAS FP16   → 4096³ 44.0 TFLOPS    TC 实际利用 ~47%（vs 94T 上限）
cuBLAS FP8e4  → 4096³ 163.8 TFLOPS   TC 实际利用 ~41%（vs 0.4 POPS 上限）
```

**结论：** 大矩阵 GEMM 不是 DRAM 瓶颈，也不是 Tensor Core 计算吞吐瓶颈——TC 实际利用在 **38~47%** 之间。核心瓶颈是 SMEM→register 的**数据供给速度**跟不上 TC 消费速率（Long SB + Short SB 合计 ~35% stall）。non-TC 指令（地址计算、同步）和寄存器压力（reg/thr=112~128, occupancy ~27%）进一步压缩了有效 TC 时间。cuBLAS 达 47% 说明供给侧已接近 Blackwell 架构上限，手写 38% 差距合理。

### 2. RMSNorm — 典型访存受限

- 算术强度 0.5 FLOP/Byte << Ridge Point 52.5
- 4096² 实测 **361 GB/s**（峰值带宽 448 GB/s 的 **81%**）
- v3 在小矩阵（512²）可达 443 GB/s（L2 命中，非纯 DRAM）

**结论：** 同一个 NCU 工具，GEMM 看 SMEM→TC stall，RMSNorm 看带宽饱和——优化方向完全不同。

### 3. Fused Conv1D+SiLU — 融合消除中间开销

- v0：5 个 kernel → 主场景 **565.9 ms**
- v2：恢复 B×L×H 并行 + 2 kernel → **53.4 ms**
- v3：CUTLASS SGEMM 投影 → **1.65 ms**，端到端 **~343×**

**结论：** 融合的价值不是让单个 kernel 更快，而是**消除中间 buffer 和 launch 开销**。

---

## 优化方法论

1. **Roofline 预判** → 算术强度判断计算/访存受限
2. **朴素 v0 基线** → 验证正确性
3. **NCU Profiling** → 定量测量 stall 原因
4. **单变量 A/B** → 每版只改一个瓶颈
5. **失败实验记录** → TileK=64、ldmatrix/swizzle 等，用数据说明 ROI

---

## 构建与测试

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build . -j$(nproc)
cd ..

# 单元测试
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_TESTS=ON
cmake --build build --target cko_unit_tests -j$(nproc)
ctest --test-dir build --output-on-failure

# 各算子 head-to-head 对比 benchmark
./build/bin/gemm_compare
./build/bin/rmsnorm_compare
```

---

## 项目结构

```
├── gemm/                  计算密集型（v0~v4 + fp16 + cuBLAS 参考）
├── rmsnorm/               访存密集型（v0~v3）
├── fused_conv1d_silu/     融合算子（v0~v3 + fused_api）
├── common/                公共工具（计时、Status、test catalog）
├── tests/                 GoogleTest（68 项：Validate + CPU ref + GPU smoke + C ABI）
├── configs/test_cases/    参数化测试用例 JSON
├── docs/testing.md        测试与 API 规范
├── .github/workflows/ci.yml  CI 门禁
└── configs/kernel_catalog.json
```

---

## 环境

| 项目 | 配置 |
|------|------|
| GPU | RTX 5060 Ti 16GB (Blackwell sm_120) |
| CUDA | 13.2 |
| FP32 CUDA Core 峰值 | 23.5 TFLOPS |
| FP16 TC 峰值 | 752 TFLOPS（实际上限 ~94T，受 SMEM→TC 供给限） |
| TF32 TC 峰值 | 188 TFLOPS（实际上限 ~24T） |
| FP8 TC 峰值 | 3.0 POPS（实际上限 ~0.4 POPS） |
| DRAM 带宽 | 448 GB/s |
