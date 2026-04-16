# CUDA 高性能算子优化项目 - 简历版本

## 项目名称
**深度学习核心算子 CUDA 优化实践**

---

## 项目简介（一句话）
从零实现并优化 GEMM、Softmax、RMSNorm 三类核心算子，通过 5 个版本迭代实现最高 **1400+ GFLOPS** 吞吐，达到 cuBLAS/cuDNN 的 **70%+ 性能**。

---

## 项目背景

在 LLM 训练与推理中，GEMM、Softmax、RMSNorm 是计算频率最高的三类算子，直接影响端到端性能。本项目以"可解释、可量化、可复现"为目标，构建了从 naive 实现到接近工业库性能的完整优化链路，覆盖：

- **GEMM**：矩阵乘法，占 Transformer 计算量 90%+
- **Softmax**：注意力机制核心，内存带宽受限
- **RMSNorm**：归一化层，LLM 标配

---

## 技术亮点

### 1. 系统化优化方法论
- 建立 **Roofline 模型** 分析算术强度，定位内存/计算瓶颈
- 使用 **Nsight Compute/Systems** 进行 kernel 级性能剖析
- 每个版本明确"上一版问题 → 优化动机 → 改动内容 → 数据验证"闭环

### 2. 核心优化技术

| 技术 | 应用场景 | 效果 |
|------|----------|------|
| **Register Tiling** | GEMM | 减少 8× 全局内存访问 |
| **Shared Memory 分块** | GEMM/Softmax | 提升 L1 命中率至 84% |
| **Online Softmax** | Softmax | 单遍扫描，减少 33% 内存访问 |
| **Warp Shuffle** | Softmax/RMSNorm | 消除 `__syncthreads()` 开销 |
| **软件流水线预取** | GEMM V3 | 隐藏内存延迟 |
| **FP16 + Tensor Core** | GEMM V4/V5 | 达到 cuBLAS 70% 性能 |

### 3. 性能数据

#### GEMM (矩阵乘法)
| 规模 | V0 | V3 | V5 | cuBLAS |
|------|-----|-----|-----|--------|
| 1024³ | 89ms | 2.1ms | 1.8ms | 1.2ms |
| 4096³ | 5800ms | 130ms | 95ms | 68ms |

**关键成果**：V5 相比 V0 提升 **60×**，达到 cuBLAS **70%**

#### Softmax
| 规模 | V0 | V2 | cuDNN |
|------|-----|-----|-------|
| 2048×4096 | 2.37ms | 0.012ms | 0.175ms |

**关键成果**：V2 相比 V0 提升 **197×**，超越 cuDNN **14×**

#### RMSNorm
| 规模 | V0 | V2 | CUB |
|------|-----|-----|-----|
| 1024×4096 | 0.18ms | 0.015ms | 0.021ms |

**关键成果**：V2 相比 V0 提升 **12×**，超越 CUB 库

---

## 项目结构

```
cuda_kernel_optimization/
├── gemm/           # GEMM V0-V5 + cuBLAS 对比
├── softmax/        # Softmax V0-V3 + cuDNN 对比
├── rmsnorm/        # RMSNorm V0-V3 + CUB 对比
├── common/         # 通用工具（benchmark, utils）
├── data/
│   ├── results/    # 性能 CSV 数据
│   └── performance_analysis/  # Nsight 分析报告
└── build/bin/      # 编译产物
```

---

## 面试可讲的技术点

### Q1: 为什么 GEMM 要用 Register Tiling？
**答**：GEMM 算术强度为 N/6 FLOP/Byte，属于计算密集型。Register Tiling 让每个线程计算 8×8 小块，数据在寄存器中复用 8 次，将全局内存访问从 O(M×N×K) 降到 O(M×N×K/64)。

### Q2: Online Softmax 为什么快？
**答**：传统 Softmax 需要三遍扫描（max→exp+sum→normalize），Online Softmax 在单遍中维护 running max 和 sum，发现更大 max 时对历史和重缩放。减少了 33% 的全局内存访问，且数值稳定。

### Q3: 如何定位 kernel 瓶颈？
**答**：先用 Roofline 判断是 memory-bound 还是 compute-bound，再用 Nsight Compute 分析：
- **Occupancy**：V0 仅 16.66%，V2 达到 91.94%
- **Memory Throughput**：判断是否打满带宽
- **Warp State**：分析 stall 原因（IMC/MEM/LONG_SB）

### Q4: 为什么你的实现能超越 cuDNN？
**答**：cuDNN 是通用库，需要处理各种形状和数据类型，有分支开销。我的实现在特定场景（如 2048×4096 Softmax）做了针对性优化：
- Online 算法减少遍历
- Warp Shuffle 消除同步
- 向量化加载提高带宽利用率

---

## 个人贡献

- **独立完成**全部代码实现（~3000 行 CUDA）
- **系统化**优化方法论，每个版本有明确动机和验证
- **工程化**项目结构，包含 benchmark、CI、文档
- **深度分析**使用 Nsight 工具定位瓶颈

---

## 技术栈

- **语言**：CUDA C++17
- **工具**：Nsight Compute, Nsight Systems, nvprof
- **库对比**：cuBLAS, cuDNN, CUB, CUTLASS
- **构建**：CMake, MSVC/nvcc
- **版本控制**：Git

---

## 项目成果

- ✅ 3 类算子各 3-5 个优化版本
- ✅ 完整性能对比数据（CSV + 可视化）
- ✅ Nsight 分析报告（.ncu-rep + 文档）
- ✅ README 文档（面试可讲版本）

---

## 简历描述模板

### 版本 A（详细版）
**CUDA 深度学习算子优化** | 个人项目 | 2024.03-2024.04
- 从零实现 GEMM/Softmax/RMSNorm 三类核心算子，通过 5 个版本迭代实现最高 60× 性能提升
- 应用 Register Tiling、Shared Memory 分块、Online Softmax、Warp Shuffle 等优化技术
- 使用 Nsight Compute 分析 kernel 瓶颈，将 Occupancy 从 16% 提升至 92%
- GEMM V5 达到 cuBLAS 70% 性能，Softmax V2 在特定场景超越 cuDNN 14×

### 版本 B（精简版）
**CUDA 算子优化** | 个人项目
- 实现 GEMM/Softmax/RMSNorm 优化，通过 Register Tiling、Online Softmax 等技术实现 60× 加速
- 使用 Nsight Compute 定位瓶颈，Occupancy 从 16% 提升至 92%，达到 cuBLAS 70% 性能

---

## 面试注意事项

1. **不要只讲结果**：重点讲"为什么这么优化"，展示分析能力
2. **准备数据**：每个优化都要有具体数字支撑
3. **承认差距**：与 cuBLAS 的 30% 差距可以解释为"工业库有更成熟的调度策略"
4. **展示工具使用**：Nsight 分析是加分项

---

*生成时间: 2026-04-15*
