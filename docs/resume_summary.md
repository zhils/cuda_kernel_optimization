# CUDA 高性能算子优化 - 简历精简版

---

## 项目一：CUDA 深度学习算子优化

**时间**：2024.03 - 2024.04  
**角色**：独立开发者

### 项目描述
从零实现并优化 GEMM、Softmax、RMSNorm 三类深度学习核心算子，通过系统化优化方法实现最高 **60× 性能提升**，达到 cuBLAS/cuDNN **70%+ 性能水平**。

### 核心工作
1. **Roofline 模型分析**：计算算术强度，定位内存/计算瓶颈，指导优化方向
2. **GEMM 优化**：应用 Register Tiling (8×8)、Shared Memory 分块、软件流水线预取，V5 版本达到 cuBLAS 70% 性能
3. **Softmax 优化**：实现 Online Softmax 单遍算法，结合 Warp Shuffle 消除同步开销，特定场景超越 cuDNN 14×
4. **性能剖析**：使用 Nsight Compute 分析 kernel 瓶颈，将 Occupancy 从 16.66% 提升至 91.94%

### 关键成果
| 算子 | 优化前 | 优化后 | 提升倍数 | 对比库 |
|------|--------|--------|----------|--------|
| GEMM (4096³) | 5800ms | 95ms | **61×** | cuBLAS 70% |
| Softmax (2048×4096) | 2.37ms | 0.012ms | **197×** | 超 cuDNN 14× |
| RMSNorm (1024×4096) | 0.18ms | 0.015ms | **12×** | 超 CUB 库 |

### 技术栈
CUDA C++17, Nsight Compute/Systems, cuBLAS, cuDNN, CMake, Git

---

## 项目亮点（面试口述版）

### 1. 系统化优化方法论
不是"试错式"优化，而是：
- 先用 Roofline 模型判断瓶颈类型
- 再用 Nsight Compute 定位具体问题（Occupancy、Memory Throughput）
- 最后针对性优化并验证

### 2. 关键技术点
- **Register Tiling**：让数据在寄存器中复用 8 次，减少全局内存访问
- **Online Softmax**：单遍扫描，减少 33% 内存访问
- **Warp Shuffle**：替代 `__syncthreads()`，消除同步开销

### 3. 工程能力
- 完整的 benchmark 框架
- 与工业库（cuBLAS/cuDNN）对比
- Nsight 分析报告文档化

---

## 面试常见问题准备

### Q: 为什么你的 Softmax 能超越 cuDNN？
**A**: cuDNN 是通用库，需要处理各种形状和数据类型，有分支判断开销。我的实现在特定场景（如大矩阵 2048×4096）做了针对性优化：
- Online 算法减少遍历次数
- Warp Shuffle 消除同步
- 向量化加载提高带宽

### Q: GEMM 为什么还差 cuBLAS 30%？
**A**: 主要差距在：
1. **调度策略**：cuBLAS 有更成熟的 grid/block 配置自适应
2. **Pipeline 深度**：工业库有更深的软件流水线
3. **Tensor Core 利用率**：我的 V5 用了 WMMA，但参数调优空间还很大

### Q: 如何定位 kernel 瓶颈？
**A**: 
1. 先算算术强度，判断 memory-bound 还是 compute-bound
2. 用 Nsight Compute 看：
   - Occupancy（V0 仅 16.66%）
   - Memory Throughput（是否打满带宽）
   - Warp State（stall 原因）
3. 针对性优化

---

## 一句话总结

**"从 naive 到接近工业库性能，用数据驱动优化，用工具定位瓶颈，展示了系统化的性能工程能力。"**
