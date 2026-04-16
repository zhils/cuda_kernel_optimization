# GEMM v0~v5 Nsight Systems 分析记录

## 1. 分析目标

用 Nsight Systems 对 `gemm_v0_naive` 到 `gemm_v5` 进行统一采样，回答两个问题：

1. 各版本在系统级时间上主要耗在哪些阶段（API 等待、Kernel、Memcpy）？
2. 从 v0 到 v5 的实现方案分别存在哪些问题，后续应如何继续优化？

---

## 2. 采集方法与环境说明

## 2.1 采集命令（核心参数）

- 工具：`Nsight Systems 2025.6.3`
- 命令形态：
  - `nsys profile --trace=cuda,nvtx --sample=none --stats=true -o <output> <exe>`
- 采集对象：
  - `gemm_v0_naive.exe`
  - `gemm_v1.exe`
  - `gemm_v2.exe`
  - `gemm_v3.exe`
  - `gemm_v4.exe`
  - `gemm_v5.exe`

## 2.2 产物位置

- `.nsys-rep/.sqlite`：
  - `build/artifacts/nsys/gemm/gemm_v0_naive.*`
  - `build/artifacts/nsys/gemm/gemm_v1.*`
  - `build/artifacts/nsys/gemm/gemm_v2.*`
  - `build/artifacts/nsys/gemm/gemm_v3.*`
  - `build/artifacts/nsys/gemm/gemm_v4.*`
  - `build/artifacts/nsys/gemm/gemm_v5.*`

## 2.3 采集注意事项

- 日志提示 `Wddm trace requires administrative privileges`，因此本次没有 WDDM/上下文切换维度。
- 每个可执行文件内部会跑多个尺寸（128~4096），因此统计是“聚合结果”，不是单尺寸点结果。

---

## 3. 核心统计汇总（Nsight Systems）

> 以下数据来自 `cuda_api_sum`、`cuda_gpu_kern_sum`、`cuda_gpu_mem_time_sum`

| 版本 | API中 `cudaEventSynchronize` 占比 | Kernel Total Time (ns) | Kernel Avg (ns) | H2D Memcpy Time (ns) | D2H Memcpy Time (ns) |
|------|-----------------------------------|------------------------|-----------------|----------------------|----------------------|
| v0 | 91.7% | 1,180,738,537 | 17,889,978 | 14,750,555 | 9,450,754 |
| v1 | 83.6% | 1,175,119,788 | 17,804,845 | 13,187,849 | 7,096,185 |
| v2 | 54.1% | 154,551,974 | 2,341,697 | 13,007,997 | 7,241,866 |
| v3 | 64.2% | 247,940,627 | 3,756,676 | 13,554,509 | 7,801,561 |
| v4 | 54.2% | 168,686,062 | 2,555,849 | 13,819,124 | 6,786,100 |
| v5 | 55.8% | 148,507,275 | 2,250,110 | 7,164,404 | 7,278,409 |

补充观察：

- v5 的 H2D 时间明显降低，和其 `half` 输入路径一致（拷贝量下降）。
- v0/v1 的系统时间大量花在 `cudaEventSynchronize`，表明 GPU 执行时长主导，CPU 端主要在等。

---

## 4. 分版本问题分析（v0~v5）

## 4.1 v0（Naive）

主要问题：

- Kernel 绝对时长最高，系统级等待（`cudaEventSynchronize`）占比极高；
- 说明内核本体效率过低，CPU 端几乎全程阻塞等待 GPU 完成。

结论：

- v0 的瓶颈不在 API 调用开销，而在 kernel 本体执行效率。

## 4.2 v1（共享内存分块）

主要问题：

- 相比 v0 没有显著降低总 kernel 时间（仍接近 1.17s 聚合量级）；
- 说明当前 v1 的 tile/线程组织并未有效突破关键瓶颈。

结论：

- 仅做共享内存分块在该实现下收益有限，需要更激进的线程级复用和计算组织。

## 4.3 v2（线程级寄存器分块）

主要问题与收益：

- kernel 总时间从 ~1.17s 降到 ~0.155s，说明方向正确；
- 但 API 中 `cudaMalloc`/`cudaMemcpy` 占比被放大（因为 kernel 变快后，固定开销更显眼）。

结论：

- v2 已把“算子内部效率”显著提升，下一步需要处理“每 case 反复分配/拷贝”的框架开销。

## 4.4 v3（寄存器预取）

主要问题：

- 相比 v2，kernel 总时间反而升高到 ~0.248s；
- 预取策略在当前参数下未形成稳定收益，可能引入寄存器压力或调度副作用。

结论：

- v3 不是稳定优于 v2 的实现，需要按尺寸或参数条件启用，而非全局替代。

## 4.5 v4（大 CTA / 更强复用）

主要问题与收益：

- kernel 总时间回落到 ~0.169s，优于 v3；
- 说明更大块级复用在大尺寸上更有效。

结论：

- v4 是对 v3 的有效修正，但仍有进一步提升空间（距离 cuBLAS 仍有差距）。

## 4.6 v5（WMMA FP16 输入 + FP32 累加）

主要问题与收益：

- kernel 总时间最低（~0.149s），系统级表现最优；
- H2D 时间显著下降（半精度输入带来的数据量收益）；
- 但结合你已有基准，v5 在超大尺寸并非总是优于 v4，说明还存在参数/调度敏感性。

结论：

- v5 是当前最有潜力的方向，但需要做“分尺寸策略 + 参数自适应”才能稳定领先。

---

## 5. 横向共性问题（跨版本）

1. **频繁分配释放开销**  
   每个 case 都在做 `cudaMalloc/cudaFree`，kernel 越快，这部分越显著。

2. **同步粒度偏粗**  
   以 `cudaEventSynchronize` 为主的串行测时，造成 CPU 长时间等待，吞吐路径不够流水化。

3. **数据搬运与计算未重叠**  
   当前是“拷贝 -> 计算 -> 拷贝回”，未见 stream 级 pipeline 叠加。

4. **版本选择缺少 runtime 策略**  
   v2/v3/v4/v5 各有优势区间，若固定单一内核，容易在某些尺寸退化。

---

## 6. 建议的下一步优化

1. **内存池化**  
   预分配 `dA/dB/dC`，避免每 case 重复 `cudaMalloc/cudaFree`。

2. **异步流水**  
   引入 stream + `cudaMemcpyAsync`，让 H2D/Kernel/D2H 形成重叠。

3. **分尺寸调度策略**  
   按 `(M,N,K)` 区间选择 v2/v4/v5，而不是固定单核。

4. **v5 参数再调优**  
   对 4096 等大尺寸单独调 block/warp/ktile，验证是否可消除 v4 反超现象。

5. **补充 Nsight Compute 联合分析**  
   Systems 定位“时间去向”，Compute 负责“SM 内部原因”（occupancy、stall、指令组合）。

---

## 7. 待补截图标记（你后续补）

- [TODO-NSYS-GEMM-1] v0 的 Timeline 全景图（展示 CPU 等待占比）
- [TODO-NSYS-GEMM-2] v2 vs v3 的 `cuda_gpu_kern_sum` 对比截图（证明 v3 回退）
- [TODO-NSYS-GEMM-3] v4 vs v5 的 memcpy 时间对比截图（展示 half 输入收益）
- [TODO-NSYS-GEMM-4] 一张跨 v0~v5 的 API 时间占比对比图（用于面试讲解）

