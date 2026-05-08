# CUDA Kernel 优化

在 **RTX 5060 Ti（Blackwell sm_120）** 上从零优化深度学习核心算子，逐版本对比 cuBLAS/cuDNN。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [gemm/README.md](gemm/README.md) | GEMM v0~v5 + FP16 + cuBLAS，从 1.42 到 37.39 TFLOPS |
| [softmax/README.md](softmax/README.md) | Softmax v0~v3 |
| [rmsnorm/README.md](rmsnorm/README.md) | RMSNorm v0~v3，带宽从 106 到 386 GB/s |
| [pytorch_extension/README.md](pytorch_extension/README.md) | PyTorch 自定义算子绑定 |

---

## 算子与版本速览

### GEMM（通用矩阵乘）

数学定义：C = A × B, A∈R^{M×K}, B∈R^{K×N}

| 版本 | 做了什么 | 4096³ TFLOPS | vs 上一版 |
|:----|----------|:-----------:|:---------:|
| v0 | 朴素三重循环 | 1.42 | — |
| v1 | 16×16 SMEM tile | 1.95 | +37% |
| v2 | 8×8/thread 寄存器分块 | 8.07 | +314% |
| v3 | cp.async + 8×4 + TileK=32 | 12.52 | +55% |
| v4 | TF32 WMMA (k=8) | 10.94 | -13% |
| v5 | WMMA+cp.async (k=8) | 11.26 | -10% |
| gemm_fp16 | FP16 WMMA (k=16) | 37.39 | — |
| cuBLAS FP32 | BF16×9 仿真 | 16.33 | — |

### Softmax

数学定义：softmax(x_i) = exp(x_i - max) / Σexp(x_j - max)

| 版本 | 做法 |
|:----|------|
| v0 | 两遍扫描，每行 1 线程 |
| v1 | SMEM staging，warp 内归约 |
| v2 | 8-warp 在线归约 |
| v3 | SMEM + warp shuffle 协同 |

### RMSNorm

数学定义：y = x / sqrt(mean(x²) + eps) × gamma

| 版本 | 做法 | 4096² 带宽 |
|:----|------|:----------:|
| v0 | 每行 1 线程串行 | 106 GB/s |
| v1 | SMEM + float4 向量化 | 386 GB/s |
| v2 | + warp shuffle 归约 | 381 GB/s |
| v3 | + weight 缓存到 SMEM | 386 GB/s |

---

## 优化流程

说穿了就一件事：找到瓶颈 → 改 → 测 → 验证。

```
理论估算（roofline）→ 朴素实现 → Nsight 看瓶颈 → 定向改动 → 对比验证 → 重复
```

具体手段：
1. Roofline：算算术强度，判断是计算还是访存受限
2. 看 PTX/SASS：哪些指令多？ld/st 还是 fma？每元素几条？
3. 看 Occupancy：register、SMEM 哪个压住了并发
4. 改一行测一次，不改假设

---

## 构建

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=120
make -j$(nproc)
cd ..

# 跑某一个版本
./build/bin/gemm_v3
./build/bin/rmsnorm_v3
```

---

## 项目结构

```
├── common/              通用基准与工具
├── gemm/                GEMM v0~v5 + fp16 + cuBLAS
├── softmax/             Softmax v0~v3
├── rmsnorm/             RMSNorm v0~v3
├── fused_conv1d_silu/   融合算子：Conv1D + SiLU
├── fused_gated_delta_rule/
├── fused_l2_norm_qk/
├── fused_output_norm_gate/
├── q_path_fusion/
├── flash_attention/     Flash Attention（v0 朴素 vs v1 tiled）
├── pytorch_extension/   PyTorch custom op 绑定
├── notes/               各专题笔记
└── CMakeLists.txt       顶层构建入口
```

---

## 环境

| 项目 | 配置 |
|------|------|
| GPU | RTX 5060 Ti 16GB |
| 架构 | Blackwell（sm_120） |
| CUDA 13.2 |
