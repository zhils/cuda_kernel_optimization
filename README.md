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
| `gemm_v0` | 97.12 ms | — | — | — | — | 无穷读全局，无复用 |
| `gemm_v1` | 70.65 ms | — | — | — | — | SMEM 分块，仍受带宽限 |
| `gemm_v2` | 17.04 ms | — | — | — | — | 寄存器分块，计算强度提升 |
| `gemm_v3` | **10.98 ms** | — | — | — | — | **12.52 TFLOPS，53% 峰值** |
| `gemm_v4` | 12.57 ms | — | — | — | 26.9% | TF32 WMMA，Occupancy 低 |
| `gemm_fp16` | **3.68 ms** | — | — | — | — | **37.39 TFLOPS** |
| `rmsnorm_v0` | 908.22 us | 0.95% | 4.55% | 12.81% | 16.48% | 基线利用率低 |
| `rmsnorm_v1` | 697.54 us | 51.31% | 39.07% | 51.31% | 8.33% | 算存均衡，SMEM 限制 occupancy |
| `rmsnorm_v2` | 321.18 us | 10.20% | 82.86% | 82.86% | 8.33% | **明显带宽受限** |
| `rmsnorm_v3` | 334.18 us | 5.88% | 86.90% | 86.90% | 38.68% | **DRAM 接近饱和** |
| `softmax_v0` | — | ~90% | ~15% | — | 高 | Memory Throughput 高 |
| `softmax_v3` | — | **84.87%** | 13.46% | — | 36.17% | **Memory Throughput 饱和** |
| `flash_attention_v0` | 599.97 us | 0.69% | 2.46% | 11.53% | 33.04% | 多 kernel 分离，利用率低 |
| `flash_attention_v1` | 705.70 us | 0.81% | 0.19% | 0.81% | 16.67% | **grid 太小，严重欠并行** |
| `flash_attention_v2` | 551.23 us | 65.09% | 0.08% | 65.09% | 27.01% | fallback kernel，算力利用中等 |
| `fused_conv1d_silu_v0` | 485.70 us | 1.70% | 0.68% | 21.87% | 16.14% | 分离路径开销 |
| `fused_conv1d_silu_v2` | 698.75 us | 4.13% | 0.56% | 90.63% | 62.08% | **片上缓存流量主导** |
| `fused_conv1d_silu_v3` | 910.14 us | 3.88% | 0.41% | 96.09% | 72.58% | **occupancy 高，瓶颈偏访存** |
| `fused_gated_delta_rule_v0` | 814.21 us | 2.28% | 31.69% | — | 16.53% | 时间维串行递推，延迟/带宽混合 |
| `fused_l2_norm_qk_v0` | 402.53 us | 29.08% | 25.20% | — | 89.22% | 算存均衡，受归约与访存共同限制 |
| `fused_output_norm_gate_v0` | 385.38 us | 11.73% | 0.49% | 95.81% | 90.97% | **L1/L2 缓存流量为主** |
| `q_path_fusion_v0` | 166.91 us | 72.45% | 4.13% | 72.45% | 88.76% | **计算占主导** |
| `q_path_fusion_v2` | 697.89 us | 17.81% | 81.28% | 81.28% | 95.80% | **RMSNorm 阶段带宽瓶颈** |

> **注：** 各算子新增版本（v1/v2 等）的 ncu profiling 可通过在本地运行 `ncu --set basic ./build/bin/<target>` 获取。

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
