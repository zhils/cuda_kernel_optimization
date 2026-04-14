# CUDA Kernel 优化

本项目逐步优化深度学习核心算子：从朴素基线实现到接近硬件峰值利用率的 CUDA Kernel，并在 **RTX 5060 Ti（Blackwell，sm_120）** 上与 cuBLAS/cuDNN 做基准对比。

## 性能总览

<!-- TODO: 在你的 GPU 上运行基准后补充 -->

| 算子 | 瓶颈 | 朴素版 → 最优版 | 相对 cuBLAS/cuDNN | 硬件峰值利用率 |
|------|------|----------------|-------------------|----------------|
| **GEMM** | 计算 | 2.48x（1024³，V0→V2） | V2 为 cuBLAS 的 3.25x 耗时（1024³） | 待补充（需 Nsight 统计） |
| **Softmax** | 内存 | _x | _% of cuDNN | _% DRAM bandwidth |
| **RMSNorm** | 内存 | _x | _% of cuDNN | _% DRAM bandwidth |

## 优化方法论

每个算子都遵循一致的系统化优化流程：

```
理论分析 → 基线实现 → 性能剖析（Nsight）→ 定向优化 → 正确性验证 → 循环迭代
```

1. **Roofline 分析** —— 计算算术强度，判断计算受限/内存受限，并给出理论性能上限
2. **朴素基线** —— 保证正确但未优化的 CUDA Kernel
3. **Nsight Compute 剖析** —— 定位真实瓶颈：warp stall、内存吞吐、占用率、指令构成
4. **渐进优化** —— 通常实现 3-5 个版本，每版聚焦一个剖析得到的瓶颈
5. **验证评估** —— 与 CPU 参考结果比对正确性，和 cuBLAS/cuDNN 比性能，报告达到理论峰值的百分比

## 算子列表

### GEMM（通用矩阵乘）— 计算受限

[详细分析 →](gemm/README.md)

| 版本 | 优化点 | 关键技术 |
|------|--------|----------|
| V0 | 朴素实现 | 每个线程计算一个输出元素 |
| V1 | 共享内存分块 | 16×16 tile，在 SMEM 中复用数据 |
| V2 | 线程级分块 | 每线程计算 4×4 输出块，寄存器复用 |
| V4 | Tensor Core | WMMA API（fp32 累加） |
| Ref | cuBLAS | `cublasSgemm` 参考实现 |

### Softmax — 内存受限

[详细分析 →](softmax/README.md)

| 版本 | 优化点 | 关键技术 |
|------|--------|----------|
| V0 | 朴素实现 | 每行单线程，3-pass |
| V1 | 共享内存归约 | block 级并行 max/sum |
| V2 | Online softmax | 单次扫描算法（Milakov 2018） |
| V3 | Warp shuffle + 向量化 | `__shfl_sync` 归约，`float4` 读取 |
| Ref | cuDNN | `cudnnSoftmaxForward` 参考实现 |

### RMSNorm — 内存受限

[详细分析 →](rmsnorm/README.md)

| 版本 | 优化点 | 关键技术 |
|------|--------|----------|
| V0 | 朴素实现 | 每行单线程 |
| V1 | Warp 归约 | 使用 `__shfl_sync` 计算 sq_sum |
| V2 | 向量化 | `float4` 读取 + 跨 warp 归约 |
| V3 | 融合实现 | 全线程写回输出，`__fmul_rn` |
| Ref | cuDNN | `cudnnNormalizationForward` 参考实现 |

## 构建与运行

```bash
# 构建全部目标
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=120
make -j$(nproc)

# 运行单个 kernel（用于 Nsight 剖析）
cd ..
./build/bin/gemm_v0_naive
./build/bin/gemm_v2_thread_tiling

# 运行完整对比基准
./build/bin/gemm_benchmark_all
./build/bin/softmax_benchmark_all
./build/bin/rmsnorm_benchmark_all

# 使用 Nsight Compute 做性能分析
ncu --set full ./build/bin/gemm_v0_naive
ncu --set full ./build/bin/gemm_v2_thread_tiling
```

## 环境信息

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB |
| 架构 | Blackwell (sm_120) |
| CUDA Toolkit | 13.2（本次实测） |
| cuBLAS / cuDNN | 最新版（随 toolkit 提供） |
| 计时方式 | `cudaEvent`，10 次迭代去极值均值，3 次 warmup |

## 项目结构

```
├── common/              通用基准工具（测试用例、计时、正确性校验）
├── gemm/                GEMM：V0–V4 + cuBLAS 参考 + benchmark
├── softmax/             Softmax：V0–V3 + cuDNN 参考 + benchmark
├── rmsnorm/             RMSNorm：V0–V3 + cuDNN 参考 + benchmark
├── docs/                环境与方法论文档
└── CMakeLists.txt       顶层构建文件
```
