# 项目简历文案 + 后续路线图 + 20道面试题

---

## 一、项目简历文案（三版）

### 版本A：300字标准版（推荐用于简历项目经历栏）

**项目名称：CUDA 高性能算子优化实践（GEMM / RMSNorm / Softmax）**

针对大模型推理中核心算子的性能瓶颈，从零构建了一条可解释、可复现的 CUDA 算子优化链路。首先通过算术强度（Arithmetic Intensity）和 Roofline 模型对三类算子进行瓶颈定性：RMSNorm（AI≈0.33）和 Softmax（AI≈0.38）为强访存受限，GEMM 小矩阵偏访存受限、大矩阵为计算受限。基于此判断，为每种算子设计了差异化优化策略。

优化过程采用 v0→v4 逐版本演进：v0 建立朴素基线；v1 引入共享内存分块与 float4 向量化加载，消除全局内存冗余访问；v2 实现线程级寄存器分块（8×8 子块），将计算密度提升 4×+；v3 采用双缓冲共享内存实现加载与计算流水线重叠；v4 集成 WMMA Tensor Core API 利用硬件矩阵乘加单元。RMSNorm 从 v0 到 v3 通过 float4 条件向量化、warp shuffle 归约和 weight 缓存，带宽利用率从 7GB/s 提升至 579GB/s（83×加速）。Softmax 自研 kernel 在多数规模下超越 cuBLAS 参考实现。

所有版本均与 cuBLAS/cuDNN 进行同精度口径对比，GEMM v3 在 4096³ 上达 13.5 TFLOPS（相对 cuBLAS 的 79%），512³ 上追平 cuBLAS。该项目形成了"理论预判→策略设计→实验验证→性能归因"的完整闭环，具备移植到 TensorRT 自定义插件或 PyTorch 自定义算子的工程基础。

**字数：约 295 字**

---

### 版本B：100字精简版（推荐用于简历摘要或个人简介）

从零构建 CUDA 算子优化链路，覆盖 GEMM、RMSNorm、Softmax 三类核心算子。基于 Roofline 模型进行瓶颈定性，采用共享内存分块、寄存器分块、双缓冲流水线和 WMMA Tensor Core 四阶段优化。RMSNorm 带宽利用率提升 83×，GEMM 在 4096³ 上达 13.5 TFLOPS，512³ 上追平 cuBLAS。所有版本具备完整的正确性验证和性能对比数据，形成可复现的"理论→策略→实验→归因"优化闭环。

**字数：约 100 字**

---

### 版本C：500字详细版（推荐用于面试自我介绍或技术博客）

**项目名称：CUDA 高性能算子优化——从朴素基线到 Tensor Core 的系统化实践**

**背景与问题：** 大模型推理中，GEMM、RMSNorm、Softmax 三类算子是计算图的绝对热点。虽然 cuBLAS/cuDNN 提供了高度优化的实现，但作为高性能算子开发工程师，必须理解"库为什么快、我怎么做到接近或超越"。本项目在 RTX 5060 Ti（Blackwell, sm_120）上，从零构建了一条完整的算子优化链路。

**方法论：** 优化前先做理论预判。计算每个算子的算术强度——GEMM 方阵为 N/6 FLOP/Byte，RMSNorm 约 0.33，Softmax 约 0.38。结合 GPU 的 FLOP/Byte 阈值（~56），判定 RMSNorm/Softmax 为强访存受限（优化主线：减少访存 + 改善访存形式），GEMM 小规模偏访存受限、大规模为计算受限（优化主线：分块复用 + 提升计算密度）。这一判断直接决定了每个版本的优化方向。

**迭代过程（v0→v4）：**

- **v0（朴素基线）：** 每线程算一个输出元素，全局内存被大量重复读取。建立正确性下限。
- **v1（共享内存分块）：** 16×16 tile + float4 向量化 + `__ldg` 只读缓存。消除全局内存冗余访问，小矩阵上已快于 cuBLAS。
- **v2（寄存器分块）：** 每线程计算 8×8 子块（64 个输出元素），CTA 覆盖 128×128。计算密度提升 4×+，512³ 上加速 3.1×。
- **v3（双缓冲流水线）：** 两组共享内存交替使用，当前 tile 计算与下一 tile 加载重叠。512³ 上再加速 2.1×（达 9.1 TFLOPS），该规模首次追平 cuBLAS。
- **v4（Tensor Core）：** 集成 WMMA API，利用硬件矩阵乘加单元。4096³ 达 13.5 TFLOPS（相对 cuBLAS 的 79%），性能差距主要来自 WMMA fragment 的额外拷贝开销和 25% occupancy。

**跨算子成果：**

- **RMSNorm（v0→v3）：** float4 向量化 + warp shuffle 归约 + weight 共享内存缓存，带宽利用率从 7 GB/s → 579 GB/s（83×），推理场景全规模超越 cuBLAS。
- **Softmax（v0→v3）：** online 算法单次遍历完成 max/sum/normalize，共享内存分块 + warp 归约，自研 kernel 多数规模优于 cuDNN。

**工程亮点：** 所有版本同精度口径对比（FP32），端到端可编译运行（CMake），benchmark 自动化（cudaEvent 计时 + CSV 导出），正确性验证（CPU golden + max_abs_diff < 1e-3）。Q_Path_Fusion 模块验证了"GEMV 不应分块、用 cuBLAS GEMM 代替"的关键认知——该决策将 4096³ 从 944ms 降至 8.66ms（108×）。

**字数：约 500 字**

---

## 二、后续需要做的 10 件事情

### 第1件：学懂双缓冲（Double Buffering）的完整原理并自己重写

**当前状态：** gemm_v3.cu 的双缓冲代码是 AI 生成的，你能跑但对核心机制理解不够深入。
**行动：**
- 手抄一遍 v3 的双缓冲代码（read_buf/write_buf 交换逻辑），逐行写出注释
- 画出 3 个 K-tile 循环的"加载→计算→交换缓冲"时间线图
- 回答三个问题：为什么需要两组而非一组 SMEM？swap 发生在 syncthreads 前还是后？为什么？
- 修改 tile 大小（如 TileK 从 16→32）观察双缓冲对 occupancy 的影响
- 输出物：一份"我手写的双缓冲 GEMM"（不从零开始，而是在 v3 基础上修改 3 处参数并解释每处的影响）

### 第2件：学懂 WMMA Tensor Core API 并手写一个最小 demo

**当前状态：** gemm_v4.cu 的 WMMA 代码是 AI 生成的。
**行动：**
- 脱离项目，写一个独立的 10 行 WMMA demo：只做一次 `mma_sync` 计算 16×16×16 的矩阵乘
- 理解 fragment 类型模板参数的含义：`matrix_a/matrix_b/accumulator`、`row_major/col_major`、`precision::tf32`
- 用 `cuobjdump -sass` 验证 mma_sync 确实生成了 HMMA 指令而非 FMA
- 对比 WMMA vs 手写 FMA 的执行时间（在 16×16×16 这个小规模上，哪个更快？为什么？）
- 输出物：`wmma_minimal_demo.cu` + 对应的 PTX/SASS 分析笔记

### 第3件：补写 Softmax 的 README.md

**当前状态：** Softmax 是唯一缺少 README 的模块。
**行动：**
- 参照 gemm/README.md 的格式，补写算术强度评估、版本演进说明、性能表格
- 跑 benchmark_all.cu 并填入实测数据
- 与 cuDNN softmax forward 做对比分析
- 特别分析：为什么 V3 online softmax 在某些规模下并不比 V2 更快？（可能因为 expf 开销占主导，online 算法省的是带宽而非计算）

### 第4件：完成 Conv1D+SiLU 的 V1 融合实现

**当前状态：** v0 基线已写好，融合方案已有（文档 02）。
**行动：**
- 把 Conv1D 的计算结果在寄存器中直接喂给 SiLU（`val/(1+expf(-val))`）
- 对比融合前（两个独立 kernel）和融合后的性能差异
- 验证理论加速比 1.84x 是否在实测中成立
- 这是你第一次从"方案设计→代码实现→性能验证"的全链路实践

### 第5件：用 Nsight Compute 完整分析 GEMM V4 并写分析报告

**当前状态：** 你有了 Nsight Compute 的使用指南（文档 10），但还没有实际跑过。
**行动：**
- 对 gemm_v4 运行 10_nsight_compute_workflow.md 中提供的完整分析脚本
- 截图关键指标（Speed Of Light 页面、Stall Reasons 饼图、Memory Workload 图表）
- 写一份 1 页的分析报告：瓶颈在哪、为什么、下一步改什么
- 特别注意：WMMA fragment load/store 的耗时占比（这能验证"WMMA extra copy overhead"的假设）

### 第6件：实现一个完整的 PyTorch 自定义算子（RMSNorm V3）

**当前状态：** 你有 PyTorch 集成指南（文档 05），代码框架已给出。
**行动：**
- 把你的 rmsnorm_v3 集成到 PyTorch（setup.py + binding C++ + CUDA kernel）
- 实现 `torch.autograd.Function` 的 backward
- 用 `torch.autograd.gradcheck` 验证梯度
- 在 Transformer 模型的一个 block 上替换 PyTorch 自带的 RMSNorm，端到端验证精度和性能

### 第7件：学习 PTX mma.sync 并对比 WMMA vs PTX mma 的性能

**当前状态：** 你知道 WMMA 有 extra copy 开销，但 PTX mma 还不会用。
**行动：**
- 阅读 CUTLASS 的 `include/cutlass/gemm/warp/mma_sm80.h`（或 SM120 对应版本）
- 写一个用 PTX inline assembly 直接调用 mma.sync 的 16×16×16 kernel
- 对比 WMMA 和 PTX mma 在相同规模下的 SASS 指令数
- 量化"WMMA extra copy"到底多了几条指令、多少 ns

### 第8件：分析 GEMM V4 低 occupancy 的根因并尝试修复

**当前状态：** 你知道 V4 occupancy ~25% 是瓶颈，但不知道精确原因。
**行动：**
- 用 Nsight Compute 查看每个 warp 的寄存器用量
- 计算：256 threads × N regs → 理论上限
- 尝试减小 tile（从 32×32→16×16）观察 occupancy 变化
- 尝试减少 WMMA fragment 数量（从每 warp 1 个 16×16 fragment → 使用更小的 K-tile）
- 目标：把 occupancy 从 25% 提到 50%+，观察是否带来实际的吞吐提升

### 第9件：为所有模块补齐"性能天花板分析"

**当前状态：** README 有性能数据，但没有"离理论上限还有多远"的分析。
**行动：**
- 对每个模块的每个版本，画出 (AI, GFLOPS) 在 Roofline 图上的位置
- 量化"性能差距 = 多少 % 天花板"
- 分解差距来源（occupancy / memory latency / bank conflict / instruction overhead）
- 写入各 README 的"天花板分析"段落

### 第10件：做一个综合 benchmark 仪表盘脚本

**当前状态：** benchmark 分散在各个 .cu 文件中，对比不方便。
**行动：**
- 写一个 Python 脚本，自动运行所有可执行文件并汇总 CSV
- 自动生成对比表格（Markdown 格式）
- 自动画出"规模 vs GFLOPS"的多版本对比折线图
- 目标是：一条命令 `python benchmark_all.py` 出全部结果 + 图表

---

## 三、20 道可能的面试题

### 基础概念类（1-5）

**1. GEMM 的算术强度怎么算？在你的 RTX 5060 Ti 上，什么规模的矩阵从 memory-bound 变为 compute-bound？**

考点：算术强度公式 N/6，FLOP/Byte 阈值 ≈ 56，N ≈ 336 是分界线。

**2. 共享内存的 bank conflict 是什么？你怎么检测和解决？**

考点：32 个 bank，同 bank 不同地址 → 串行化；+4 padding 是最简单解法；用 Nsight Compute metrics 验证。

**3. `__ldg`、`__ldcs`、普通 load 的区别是什么？各适用于什么场景？**

考点：__ldg→只读缓存、__ldcs→绕过 L1 走 streaming、普通→L1+L2。数据有复用用 __ldg，只用一次用 __ldcs。

**4. warp shuffle 和共享内存归约的区别？什么时候用哪个？**

考点：shuffle 不需要 SMEM、不需要 syncthreads、只在 warp 内有效；跨 warp 仍需 SMEM。RMSNorm V3 先用 shuffle 在 warp 内归约，再用 SMEM 跨 warp。

**5. occupancy 是什么？为什么重要？你的 GEMM V4 occupancy 为什么低？**

考点：活跃 warp 数 / 理论最大 warp 数。受寄存器、SMEM、block size 影响。V4 的 WMMA fragment 占用大量寄存器 → occupancy 25%。

### 优化策略类（6-10）

**6. 给你一个从未优化过的 GEMM kernel，你的优化步骤是什么？**

考点：先定性（算 AI → 判断 memory/compute-bound）→ 再定量（v0 基线 → v1 SMEM → v2 寄存器分块 → v3 双缓冲/异步 → v4 Tensor Core）。每一步都要有数据支撑。

**7. 为什么 GEMV 不应该用分块（tiling）？你项目中的哪个例子证明了这一点？**

考点：GEMV 数据复用次数 R=1，tile 无复用收益但有 syncthreads 开销。q_path_fusion_v2 初版在 4096×4096 上比 v1 慢 12.7×，改用 cuBLAS GEMM 从 944ms→8.66ms。

**8. cp.async 比普通同步加载好在哪里？什么时候用 cp.async 没有收益？**

考点：加载和计算重叠；K-tile ≤ 2 或极强 compute-bound 时无收益。你的 V3 在 512³ 上加速 2.1×（最强），4096³ 上加速 1.2×（减弱）。

**9. 算子融合的收益来自哪里？你怎么量化融合前后的提升？**

考点：消除中间结果的 HBM 往返（写+读=2×）。公式：加速比 ≈ 未融合访存量 / 融合后访存量。Conv1D+SiLU 预期 1.84x。

**10. 你的 GEMM V4 用了 WMMA，为什么还是比 cuBLAS 慢？**

考点：WMMA fragment 有额外 copy 开销 + occupancy 只有 25%。cuBLAS 内部用 PTX mma.sync（零 copy）+ 更好的 warp 调度。差距 ~21%。

### 编程与工具类（11-15）

**11. 怎么写一个 PyTorch 自定义 CUDA 算子？从 .cu 文件到 `torch.ops.xxx` 的完整流程。**

考点：TORCH_LIBRARY 注册 → TORCH_LIBRARY_IMPL 绑定 CUDA → autograd.Function 支持求导 → setup.py 编译。

**12. 你用 cuobjdump 看过生成的 PTX/SASS 吗？怎么确认 Tensor Core 真的被使用了？**

考点：`cuobjdump -ptx` 搜 `mma.sync`，`cuobjdump -sass` 搜 `HMMA`。

**13. Nsight Compute 中你最关注的 5 个指标是什么？**

考点：SM Throughput、Memory Throughput、Occupancy、Top Stall Reasons、Bank Conflicts。

**14. 怎么测试一个 CUDA kernel 的正确性？**

考点：CPU golden 对比（同精度）、max_abs_diff < 1e-3、多种规模（对齐/非对齐）、边界条件。

**15. 如果 kernel 中出现了 `st.local`（local memory store），说明什么？怎么解决？**

考点：寄存器溢出到 HBM，极慢。解法：减小 tile size、减少每线程寄存器用量、用 `__launch_bounds__` 提示编译器。

### 架构与深度类（16-20）

**16. 你的代码在 A100 / H100 / RTX 4090 上跑，需要改什么？**

考点：SMEM 大小（不同架构）、Tensor Core 指令格式（sm80 vs sm90 vs sm120）、cp.async 可用性（sm80+）、TMA（sm90+ exclusive）。最核心的差异是不同 FLOP/Byte 阈值影响 tile size 选择。

**17. INT8 量化 GEMM 怎么做？`__dp4a` 怎么用？精度损失怎么补偿？**

考点：4 个 INT8 打包到 int32 → __dp4a 一次做 4 次乘加 → INT32 累加。补偿：per-channel scaling、SmoothQuant、混合精度各层分配。

**18. 如果让你从零实现 FlashAttention，你会怎么设计？**

考点：分块计算 QK^T + online softmax（避免存储完整 attention matrix → O(N²)→O(N) 显存）。核心难点：online rescaling 的数值稳定性，SRAM/DRAM 数据流编排。

**19. 你的 RMSNorm V3 在 4096 列上 579 GB/s 的带宽利用率意味着什么？**

考点：超过了 RTX 5060 Ti 的理论 DRAM 带宽（~448 GB/s），说明部分数据来自 L2 cache 命中。这是 memory-bound 算子接近硬件极限的标志——再优化的空间很小。

**20. 如果你接手一个陌生 GPU 架构（如 AMD MI300X），你怎么快速让你的 kernel 在上面跑起来？**

考点：先用 hipify 工具转换 CUDA→HIP；最关键的是处理 warp(32)→wavefront(64) 的差异（warp shuffle 的 offset、tile size 重新计算）；Tensor Core → MFMA 指令对应；用 roofline 重新计算转折点以决策 tile 参数。
