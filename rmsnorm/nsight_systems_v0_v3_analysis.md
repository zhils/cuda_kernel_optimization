# RMSNorm v0~v3 Nsight Systems 分析记录

## 1. 分析目标

基于 Nsight Systems 对 `rmsnorm_v0_naive`、`rmsnorm_v1`、`rmsnorm_v2`、`rmsnorm_v3` 做统一 profile，回答：

1. 不同实现方案在系统级（API/Kernel/Memcpy）时间分布上有什么差异？
2. 各版本当前主要问题是什么？
3. 下一步应该优先优化哪里？

---

## 2. 采集方式

- 工具：`Nsight Systems 2025.6.3`
- 命令形式：`nsys profile --trace=cuda,nvtx --sample=none --stats=true`
- 采集对象：
  - `rmsnorm_v0_naive.exe`
  - `rmsnorm_v1.exe`
  - `rmsnorm_v2.exe`
  - `rmsnorm_v3.exe`

输出目录：

- `build/artifacts/nsys/rmsnorm/`
  - `rmsnorm_v0_naive.nsys-rep/.sqlite`
  - `rmsnorm_v1.nsys-rep/.sqlite`
  - `rmsnorm_v2.nsys-rep/.sqlite`
  - `rmsnorm_v3.nsys-rep/.sqlite`

> 说明：日志提示 `CPU context switches trace requires administrative privileges`，本次不含上下文切换统计。

---

## 3. 核心结果（来自 nsys stats）

## 3.1 CUDA API 时间分布（主要项）

| 版本 | `cudaMalloc` 占比 | `cudaMemcpy` 占比 | `cudaEventSynchronize` 占比 | 备注 |
|------|-------------------|-------------------|-----------------------------|------|
| v0 | 60.5% | 12.5% | 1.5% | `cuLibraryLoadData` 占 23.6%（一次性加载） |
| v1 | 36.8% | 10.2% | 2.3% | `cuLibraryLoadData` 占 48.5%（一次性加载） |
| v2 | 71.8% | 16.1% | 3.5% | API 端主要耗在反复分配释放 |
| v3 | 71.0% | 15.6% | 3.7% | API 端模式与 v2 接近 |

## 3.2 Kernel 汇总

| 版本 | Kernel 名称 | Instances | Kernel Total Time (ns) | Kernel Avg (ns) |
|------|-------------|-----------|------------------------|-----------------|
| v0 | `RMSNormKernel` | 5 | 1,825,946 | 365,189 |
| v1 | `RMSNormV1StagedKernel` | 55 | 4,440,624 | 80,739 |
| v2 | `RMSNormV2StagedKernel` | 55 | 3,952,892 | 71,871 |
| v3 | `RMSNormV3StagedKernel` | 55 | 4,446,193 | 80,840 |

## 3.3 Memcpy 时间汇总

| 版本 | H2D Total (ns) | D2H Total (ns) | 主要特点 |
|------|----------------|----------------|----------|
| v0 | 6,392,958 | 5,362,716 | 与其他版本同量级 |
| v1 | 7,405,601 | 7,090,975 | 拷贝占比上升 |
| v2 | 6,563,503 | 5,476,401 | 拷贝比 v1 略好 |
| v3 | 5,786,612 | 7,083,039 | D2H 占比偏高 |

---

## 4. 分版本问题分析

## 4.1 v0（Naive）

主要问题：

- Kernel 单次耗时高（平均 365us），说明核心计算路径效率低；
- 行内串行特征明显，GPU 计算吞吐受限；
- 运行流程中分配/拷贝仍是可见开销，但不是主瓶颈。

结论：

- v0 的核心问题在 kernel 本体，符合“单线程逐行”实现预期。

## 4.2 v1（Staged）

主要问题：

- 虽然单 kernel 平均耗时从 365us 降到 80us，说明内核优化有效；
- 但 API 侧开销（特别是 `cudaMalloc/cudaFree`）开始成为系统级显性瓶颈；
- 表明“算子快了，但框架调度方式没跟上”。

结论：

- v1 的问题不再只在计算逻辑，而是“每 case 重复分配/释放”的执行框架问题。

## 4.3 v2（Vectorized/Warp 归约）

主要问题与收益：

- kernel 平均时间进一步降到 71.9us（四版中最低）；
- 但 API 中 `cudaMalloc` 占比最高（71.8%），系统瓶颈明显转移到分配路径。

结论：

- v2 是当前计算内核效率最优版本之一，但整体端到端效率被内存管理开销拖住。

## 4.4 v3（Fused Path）

主要问题：

- kernel 平均时间回升到 80.8us（接近 v1），较 v2 有退化；
- D2H 总时间偏高，说明输出阶段或数据路径存在额外代价。

结论：

- v3 并非全场景优于 v2；当前实现下，v2 仍是更稳健的高性能选择。

---

## 5. 跨版本共性问题

1. **重复分配释放开销偏大**  
   `cudaMalloc/cudaFree` 在 v1~v3 都占据了明显 API 时间，尤其 v2/v3。

2. **系统级瓶颈发生迁移**  
   从 v0 的“kernel 慢”迁移到 v2/v3 的“框架开销（分配/拷贝）显性化”。

3. **Memcpy 未与计算重叠**  
   仍是典型 `H2D -> Kernel -> D2H` 串行结构，没有异步 pipeline。

---

## 6. 优化建议（按优先级）

1. **引入预分配/内存池**  
   将 `cudaMalloc/cudaFree` 从每 case 执行改为一次分配复用。

2. **异步流水化**  
   使用 stream + `cudaMemcpyAsync` 尝试重叠拷贝与计算。

3. **版本选择策略**  
   默认优先 v2；v3 仅在验证后对特定尺寸启用。

4. **补充 Nsight Compute 联合分析**  
   Systems 定位“时间花在哪”；Compute 解释“SM 内部为何慢”。

---

## 7. 待补截图标记（你后续补）

- [TODO-NSYS-RMS-1] v0 timeline（展示 kernel 单次耗时显著）
- [TODO-NSYS-RMS-2] v1/v2 API 占比对比图（突出 `cudaMalloc`）
- [TODO-NSYS-RMS-3] v2 vs v3 的 kernel summary 对比图
- [TODO-NSYS-RMS-4] memcpy 时间对比图（H2D/D2H 分离）

