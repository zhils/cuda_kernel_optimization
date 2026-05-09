# CUDA Kernel 优化

在 **RTX 5060 Ti（Blackwell sm_120）** 上从零实现并优化深度学习核心算子，逐版本对比 NVIDIA 官方库（cuBLAS / cuDNN）。

---

## 算子一览

| 算子 | 目录 | 优化版本数 | 关键成果 |
|------|------|:----------:|----------|
| GEMM | [gemm/](gemm/README.md) | 6（v0~v5 + fp16） | FP32 CUDA Core 达 12.52 TFLOPS（峰值 53%），FP16 Tensor Core 达 37.39 TFLOPS |
| Softmax | [softmax/](softmax/README.md) | 4（v0~v3） | Online 单遍算法 + float4 向量化 + Warp Shuffle，Memory Throughput 84.87% |
| RMSNorm | [rmsnorm/](rmsnorm/README.md) | 4（v0~v3） | 带宽从 106 GB/s 提升至 386 GB/s（GDDR7 理论带宽 86%） |
| Fused Conv1D+SiLU | [fused_conv1d_silu/](fused_conv1d_silu/README.md) | 4（v0~v3） | 端到端 2.6× 加速（5 kernel → 2 kernel 融合） |
| Flash Attention | [flash_attention/](flash_attention/README.md) | 5（v0~v4） | v1 Tiled Online-Softmax，v3 2D Grid 消除外层循环提升 Occupancy 至 ~60% |
| Fused Gated Delta Rule | [fused_gated_delta_rule/](fused_gated_delta_rule/README.md) | 3（v0~v2） | v1 全融合消除中间缓冲，v2 双 head ILP + float4 FMA 权重加载 |
| Fused L2 Norm Q/K | [fused_l2_norm_qk/](fused_l2_norm_qk/README.md) | 3（v0~v2） | v1 3D grid Q/K 融合 + Warp Shuffle，v2 2 行/block + 4 路 ILP |
| Fused Output Norm Gate | [fused_output_norm_gate/](fused_output_norm_gate/README.md) | 3（v0~v2） | v1 单 kernel 全融合消除 3 个中间缓冲，v2 2 行/block 权重复用 |
| Q Path Fusion | [q_path_fusion/](q_path_fusion/README.md) | 3（v0~v2） | RMSNorm + Linear(Q) 融合，v2 达 95.80% Occupancy |
| PyTorch Extension | [pytorch_extension/](pytorch_extension/README.md) | — | Softmax + RMSNorm 的 PyTorch 自定义算子绑定 |

---

## 优化方法论

每个算子的优化遵循 **"找到瓶颈 → 定向改动 → A/B 对比 → 验证"** 的闭环：

1. **Roofline 预判**：估算算术强度，判断访存受限还是计算受限
2. **朴素实现**：验证正确性的基线版本
3. **Nsight Compute Profiling**：定量测量 Memory Throughput、Compute Throughput、Occupancy、Stall Reasons
4. **PTX/SASS 分析**：检查寄存器溢出、FMA 使用、循环展开等编译器行为
5. **单变量 A/B 测试**：每版只改一个瓶颈，量化对比收益

---

## Nsight Compute 瓶颈总览

命令：`ncu --set basic --target-processes all --kernel-name-base demangled`。  
统计口径：每个可执行文件取 Duration 最大的一次 kernel launch。

| 目标 | Max Duration(us) | Compute(SM) | DRAM | Memory | Achieved Occupancy | 瓶颈分析 |
|:-----|-----------------:|------------:|-----:|-------:|-------------------:|:---------|
| `gemm_v0` | 945.57 ms | 35.30%¹ | 4.39%¹ | 36.13%¹ | 99.84% | 无穷读全局，无复用 |
| `gemm_v1` | 710.50 ms | 33.06%¹ | 5.24%¹ | 35.38%¹ | 99.83% | SMEM 分块，仍受带宽限 |
| `gemm_v2` | 219.79 ms | 0.93%¹ | 1.81%¹ | 50.25%¹ | 16.66% | 寄存器分块，计算强度提升 |
| `gemm_v3` | **177.69 ms** | 1.61%¹ | 3.56%¹ | 43.68%¹ | 33.07% | **12.52 TFLOPS，53% 峰值** |
| `gemm_v4` | 180.33 ms | 1.81%¹ | 1.91%¹ | 31.72%¹ | 26.90% | TF32 WMMA，Occupancy 低 |
| `gemm_fp16` | **102.81 ms** | 0.78%¹ | 1.89%¹ | 50.83%¹ | 32.93% | **37.39 TFLOPS** |
| `rmsnorm_v0` | 908.22 us | 0.95% | 4.55% | 12.81% | 16.48% | 基线利用率低 |
| `rmsnorm_v1` | 697.54 us | 51.31% | 39.07% | 51.31% | 8.33% | 算存均衡，SMEM 限制 occupancy |
| `rmsnorm_v2` | 321.18 us | 10.20% | 82.86% | 82.86% | 8.33% | **明显带宽受限** |
| `rmsnorm_v3` | 334.18 us | 5.88% | 86.90% | 86.90% | 38.68% | **DRAM 接近饱和** |
| `softmax_v0` | 647.65 | 0.69% | 2.78% | 12.30% | 16.64% | 朴素版本低效 |
| `softmax_v1` | 86.02 | **79.99%** | 11.74% | 79.99% | 33.55% | **算存利用均高** |
| `softmax_v2` | 340.06 | 16.02% | **84.51%** | **84.51%** | 8.31% | **带宽受限明显** |
| `softmax_v3` | 338.94 | 12.96% | **84.86%** | **84.86%** | 8.31% | **DRAM 饱和** |
| `flash_attention_v0` | 599.97 | 0.69% | 2.46% | 11.53% | 33.04% | 多 kernel 分离，利用率低 |
| `flash_attention_v1` | 705.70 | 0.81% | 0.19% | 0.81% | 16.67% | **grid 太小，严重欠并行** |
| `flash_attention_v2` | 551.23 | 65.09% | 0.08% | 65.09% | 27.01% | fallback kernel，算力利用中等 |
| `flash_attention_v3` | 76.28² | 1.07%² | 0.82%² | 1.06%² | 33.30%⁵ | 2D Grid 设计，Waves/SM 偏小 |
| `flash_attention_v4` | 76.09² | 1.30%² | 0.67%² | 1.36%² | 44.83%⁵ | Bank-free + ILP，小规模测试 |
| `fused_conv1d_silu_v0` | 485.70 us | 1.70% | 0.68% | 21.87% | 16.14% | 分离路径开销 |
| `fused_conv1d_silu_v2` | 698.75 us | 4.13% | 0.56% | 90.63% | 62.08% | **片上缓存流量主导** |
| `fused_conv1d_silu_v3` | 910.14 us | 3.88% | 0.41% | 96.09% | 72.58% | **occupancy 高，瓶颈偏访存** |
| `fused_gated_delta_rule_v0` | 814.21 us | 2.28% | 31.69% | 31.69% | 16.53% | 时间维串行递推，延迟/带宽混合 |
| `fused_gated_delta_rule_v1` | 368.17 us³ | 1.70%³ | 0.53%³ | 22.08%³ | 16.67% | 全融合消除中间缓冲 |
| `fused_gated_delta_rule_v2` | 145.94 us³ | 0.09%³ | 0.61%³ | 29.22%³ | 8.33% | 双 head ILP + FMA |
| `fused_l2_norm_qk_v0` | 402.53 us | 29.08% | 25.20% | 29.08% | 89.22% | 算存均衡，受归约与访存共同限制 |
| `fused_l2_norm_qk_v1` | 225.24 us | 20.76% | 4.31% | 11.81% | 67.13% | 3D grid Q/K 融合 + Warp Shuffle |
| `fused_l2_norm_qk_v2` | 238.10 us | 12.96% | 5.34% | 10.59% | 55.21% | 2 rows/block + 4 路 ILP |
| `fused_output_norm_gate_v0` | 385.38 us | 11.73% | 0.49% | 95.81% | 90.97% | **L1/L2 缓存流量为主** |
| `fused_output_norm_gate_v1` | 11.54 us⁴ | 7.29%⁴ | 1.58%⁴ | 51.09%⁴ | 99.19%⁵ | 全融合消除 3 中间缓冲 |
| `fused_output_norm_gate_v2` | 8.37 us⁴ | 6.25%⁴ | 2.35%⁴ | 19.81%⁴ | 82.66%⁵ | 2 rows/block 权重复用 |
| `q_path_fusion_v0` | 166.91 us | 72.45% | 4.13% | 72.45% | 88.76% | **计算占主导** |
| `q_path_fusion_v2` | 697.89 us | 17.81% | 81.28% | 81.28% | 95.80% | **RMSNorm 阶段带宽瓶颈** |

> **注：**
> ¹ GEMM Compute/DRAM/Memory 数据取自 128³ 小规模 ncu 指标。实际大规摸性能见各版本 TFLOPS 数据。
> ² flash_attention v3/v4 为 N=64 小规模测试数据（grid 仅 2 blocks）。
> ³ fused_gated_delta_rule v1 为 B=4,L=1024,D=512,H=256 配置；v2 为 B=8,L=2048,D=512,H=256 配置。
> ⁴ fused_output_norm_gate v1/v2 为 B=128 小规模配置数据。
> ⁵ Occupancy 取该版本最大规模 launch 配置的数据。
> 各算子新增版本的详细 ncu profiling 可通过 `ncu --set basic ./build/bin/<target>` 在本地获取。

---

## Warp Stall 原因分析

命令：`ncu --metrics smsp__average_warps_issue_stalled_*_per_issue_active.ratio`。  
统计口径：取该版本 Duration 最大的一次 launch 的 Top 5 stall 原因（百分比按归一化后的相对占比）。

| 目标 | #1 Stall | #2 Stall | #3 Stall | #4 Stall | #5 Stall |
|:-----|:---------|:---------|:---------|:---------|:---------|
| `gemm_v0` | Long Scoreboard 79.8% | Not Selected 12.0% | Wait 7.9% | No Instruction 0.2% | Math Pipe 0.1% |
| `gemm_v1` | Mio Throttle 43.5% | Long Scoreboard 40.2% | Not Selected 7.6% | Wait 5.6% | Short Scoreboard 2.7% |
| `gemm_v2` | Long Scoreboard 52.5% | Short Scoreboard 20.5% | Not Selected 16.3% | Wait 6.5% | Mio Throttle 2.2% |
| `gemm_v3` | Long Scoreboard 46.4% | Not Selected 25.8% | Short Scoreboard 13.1% | Mio Throttle 6.9% | Wait 5.8% |
| `gemm_v4` | **Math Pipe Throttle 72.9%** | Wait 15.7% | Long Scoreboard 6.5% | Not Selected 2.7% | Short Scoreboard 0.9% |
| `gemm_fp16` | Long Scoreboard 46.8% | Short Scoreboard 20.2% | Math Pipe Throttle 16.1% | Wait 9.1% | Mio Throttle 6.7% |
| `rmsnorm_v0` | **Long Scoreboard 97.4%** | Wait 2.5% | No Instruction 0.0% | Not Selected 0.0% | Short Scoreboard 0.0% |
| `rmsnorm_v1` | Wait 34.7% | Long Scoreboard 31.0% | Short Scoreboard 21.3% | Mio Throttle 11.8% | No Instruction 1.2% |
| `rmsnorm_v2` | **Long Scoreboard 81.8%** | Wait 8.7% | Short Scoreboard 8.4% | Mio Throttle 0.7% | No Instruction 0.4% |
| `rmsnorm_v3` | **Long Scoreboard 78.7%** | Mio Throttle 11.0% | Short Scoreboard 8.8% | Wait 1.2% | Not Selected 0.2% |
| `softmax_v0` | **Long Scoreboard 97.4%** | Wait 1.4% | Short Scoreboard 1.1% | No Instruction 0.1% | — |
| `softmax_v1` | **Wait 49.3%** | Short Scoreboard 32.1% | Long Scoreboard 10.7% | Mio Throttle 6.5% | No Instruction 1.4% |
| `softmax_v2` | **Long Scoreboard 73.6%** | Wait 13.0% | Short Scoreboard 11.7% | No Instruction 0.9% | Mio Throttle 0.9% |
| `softmax_v3` | **Long Scoreboard 79.1%** | Wait 13.2% | Short Scoreboard 6.4% | No Instruction 0.9% | Math Pipe 0.2% |
| `flash_attention_v0` | Short Scoreboard 38.9% | Long Scoreboard 31.7% | Wait 26.9% | No Instruction 1.2% | Not Selected 1.0% |
| `flash_attention_v1` | Long Scoreboard 52.7% | Not Selected 27.1% | Wait 18.3% | Short Scoreboard 1.2% | No Instruction 0.6% |
| `flash_attention_v2` | Long Scoreboard 52.7% | Not Selected 27.1% | Wait 18.3% | Short Scoreboard 1.2% | No Instruction 0.6% |
| `flash_attention_v3` | Short Scoreboard 40.7% | Wait 33.8% | No Instruction 15.0% | Not Selected 6.2% | Long Scoreboard 2.3% |
| `flash_attention_v4` | Long Scoreboard 47.2% | Short Scoreboard 31.6% | Wait 16.7% | Not Selected 3.7% | Math Pipe 0.3% |
| `fused_conv1d_silu_v0` | **Long Scoreboard 92.0%** | Wait 4.5% | Short Scoreboard 2.1% | Not Selected 0.7% | Math Pipe 0.4% |
| `fused_conv1d_silu_v1` | **Long Scoreboard 99.0%** | Wait 0.9% | No Instruction 0.0% | Short Scoreboard 0.0% | — |
| `fused_conv1d_silu_v2` | Long Scoreboard 58.6% | Wait 18.2% | Not Selected 9.9% | Short Scoreboard 8.3% | Math Pipe 3.2% |
| `fused_conv1d_silu_v3` | **Long Scoreboard 69.3%** | Wait 15.7% | Short Scoreboard 6.0% | Not Selected 5.3% | Mio Throttle 2.2% |
| `fused_gated_delta_rule_v0` | **Long Scoreboard 86.3%** | Wait 10.8% | Short Scoreboard 2.0% | Not Selected 0.4% | Math Pipe 0.3% |
| `fused_gated_delta_rule_v1` | **Long Scoreboard 96.7%** | Wait 3.0% | No Instruction 0.1% | Short Scoreboard 0.1% | Math Pipe 0.1% |
| `fused_gated_delta_rule_v2` | **Long Scoreboard 96.6%** | Wait 2.9% | Short Scoreboard 0.4% | No Instruction 0.1% | — |
| `fused_l2_norm_qk_v0` | Long Scoreboard 38.1% | Short Scoreboard 26.4% | Wait 20.7% | Not Selected 7.0% | Mio Throttle 5.2% |
| `fused_l2_norm_qk_v1` | Long Scoreboard 46.1% | Wait 24.2% | Short Scoreboard 17.0% | Not Selected 9.0% | Math Pipe 1.5% |
| `fused_l2_norm_qk_v2` | Long Scoreboard 35.4% | Short Scoreboard 27.9% | Wait 25.0% | No Instruction 5.6% | Not Selected 5.5% |
| `fused_output_norm_gate_v0` | Long Scoreboard 38.5% | Short Scoreboard 24.8% | Wait 20.1% | Not Selected 8.6% | Mio Throttle 5.2% |
| `fused_output_norm_gate_v1` | **Mio Throttle 73.7%** | Long Scoreboard 18.9% | Short Scoreboard 5.0% | Not Selected 1.3% | Wait 1.0% |
| `fused_output_norm_gate_v2` | **Mio Throttle 50.0%** | Long Scoreboard 33.3% | Short Scoreboard 14.7% | Wait 1.1% | Not Selected 0.8% |
| `q_path_fusion_v0` | **Long Scoreboard 79.7%** | Mio Throttle 7.9% | Wait 5.5% | Short Scoreboard 3.9% | Not Selected 2.7% |
| `q_path_fusion_v2` | Mio Throttle 27.2% | Short Scoreboard 26.8% | Long Scoreboard 15.4% | Not Selected 12.9% | No Instruction 8.5% |

**关键发现：**
- **Long Scoreboard**（等待全局内存/纹理加载完成）是占主导的 stall 原因，尤其在 memory-bound 的 kernel（softmax_v0/v2/v3、rmsnorm_v0/v2/v3）中占 70-97%。
- **Mio Throttle**（内存 I/O 管道瓶颈）在融合 kernel（fused_output_norm_gate_v1/v2）中显著，因其 SMEM 读取密集。
- **Math Pipe Throttle**（计算管道瓶颈）在 TF32 WMMA 的 gemm_v4 中占 72.9%，说明 Tensor Core 算力已达硬件上限。
- **Wait**（等待其他 warp 完成）在 softmax_v1 中占 49.3%，因其 warp shuffle 归约阶段需跨 warp 同步。
- **Short Scoreboard**（等待 L1/L2 缓存）在 flash_attention_v0/v3 中较高，因小规模测试时数据在片上缓存中。

---

## 构建与运行

**环境要求：** CUDA 13.2+，Compute Capability sm_120（Blackwell）

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=120
make -j$(nproc)
cd ..

# 运行单个 kernel
./build/bin/gemm_v3
./build/bin/softmax_v3
./build/bin/rmsnorm_v3

# 统一 Benchmark
./build/bin/gemm_fp16
./build/bin/softmax_cudnn_ref
```

---

## 项目结构

```
├── CMakeLists.txt                 顶层构建
├── common/                       公共工具（CUDA 宏、计时器、矩阵工具）
│   ├── include/common/
│   │   ├── benchmark.h
│   │   └── cuda_utils.h
│   └── src/benchmark.cpp
│
├── gemm/                         通用矩阵乘（v0~v5 + fp16 + int8 + cuBLAS/cuBLASLt）
│   ├── README.md                 性能数据、Nsight 分析
│   ├── gemm_v0~v5.cu            CUDA Core / WMMA / WGMMA 各版本
│   ├── gemm_fp16.cu              FP16 Tensor Core
│   ├── gemm_int8.cu              INT8 WMMA 量化
│   ├── gemm_fp8_cublaslt.cu      FP8 cuBLASLt
│   └── quant_gemm_compare.cu     低精度舍入对比
│
├── softmax/                      Softmax（v0~v3 + cuDNN 参考）
│   ├── README.md
│   ├── softmax_v0~v3.cu
│   ├── softmax_kernels.cuh
│   └── benchmark_all.cu          统一 benchmark
│
├── rmsnorm/                      RMSNorm（v0~v3）
│   ├── README.md                 Nsight 瓶颈分析
│   ├── rmsnorm_v0~v3.cu
│   └── rmsnorm_kernels.cuh
│
├── flash_attention/              Flash Attention（v0~v4）
│   ├── README.md                 各版本架构演进对比
│   ├── flash_attention_v0~v4.cu
│   └── flash_attn_v2_blackwell.md
│
├── fused_conv1d_silu/            融合 Conv1D + SiLU（v0~v3）
│   └── ...
│
├── fused_gated_delta_rule/       融合 Gated Delta Rule（v0~v2）
│   └── ...
│
├── fused_l2_norm_qk/             融合 L2 Norm Q/K（v0~v2）
│   └── ...
│
├── fused_output_norm_gate/       融合 Output Norm Gate（v0~v2）
│   └── ...
│
├── q_path_fusion/                Q 路径融合 RMSNorm + Linear（v0~v2）
│   └── ...
│
├── pytorch_extension/            PyTorch 自定义算子绑定
│   ├── setup.py
│   ├── binding.cpp
│   └── test_ops.py
│
├── gemm/quantization_fp16_fp8_int8.md  低精度量化数据汇总
└── LICENSE                               Apache 2.0
```

---

## 环境

| 项目 | 配置 |
|------|------|
| GPU | RTX 5060 Ti 16GB |
| 架构 | Blackwell（sm_120） |
| 驱动 | CUDA 13.2 |
| FP32 峰值 | 23.5 TFLOPS（36 SM × 128 Core × 2.55 GHz × 2） |
| FP16 TC 峰值 | 376 TFLOPS（36 SM × 4096 × 2.55 GHz） |
| 显存带宽 | 448 GB/s（GDDR7 × 128-bit） |
