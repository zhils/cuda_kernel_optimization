# GEMM — 通用矩阵乘

## 数学定义

```
C[m, n] = Σ_{k=0}^{K-1} A[m, k] × B[k, n]

A ∈ R^{M×K},  B ∈ R^{K×N},  C ∈ R^{M×N}
```

## 理论性能分析

### 算术强度


| 指标    | 公式                          | 数值（方阵 N×N）        |
| ----- | --------------------------- | ----------------- |
| 数据搬运量 | (M×K + K×N + M×N) × 4 bytes | 12N² bytes        |
| 计算量   | 2 × M × N × K FLOPs         | 2N³ FLOPs         |
| 算术强度  | 2N³ / 12N²                  | **N/6 FLOP/Byte** |


### Roofline 分类

RTX 5060 Ti 规格：

- FP32 峰值算力：~25 TFLOPS
- DRAM 带宽：~448 GB/s
- 平衡点（ridge point）：~55 FLOP/Byte


| 矩阵大小（N×N） | 算术强度 | 分类       |
| --------- | ---- | -------- |
| 128       | ~21  | **访存受限** |
| 512       | ~85  | **计算受限** |
| 1024      | ~170 | **计算受限** |
| 4096      | ~683 | **计算受限** |


对于典型深度学习尺寸（N > 256），GEMM 属于 **计算受限**。（硬件实际性能距离理论性能有些许差距，基本上可以认为40+就会到平衡点，从 访存受限 变为 计算受限。
优化重点应放在提升 FMA 利用率并减少 warp stall。

## 优化版本

### V0：朴素基线

**文件：** `gemm_v0_naive.cu`

每个线程计算 C 的一个元素，从全局内存读取 A 的整行和 B 的整列。

- **问题：** 全局内存重复读取严重——每个 A/B 元素会被读取 M 或 N 次。此时计算强度 = 1/2*4=0.125(1次乘加，2次数据读取)，属于 访存受限，远远不及理论上的 计算受限。
- **Nsight 诊断：** warp stall 主要由 `Stall Long Scoreboard`（内存延迟）和 `Stall LG Throttle`（LSU 饱和）主导
  Nsight的性能图可以参考图gemm_v0_naive.png，可以从图中看到stall中占比前3的分别是：
  a. stall LG Throttle:访存拥塞：因内存子系统拥塞，内存指令（LDG）被节流，无法向LSU（Load/Store Unit）发射。
  b. Stall Long Scoreboard: 等待长延迟数据：等待一个离开SM的长延迟操作结果（如全局内存读取），数据还未返回。
  c. Stall Not selected: 等待调度：指令和依赖都已就绪，但当前周期没被调度器选中，在排队等待。

### V1：共享内存分块（16×16）

**文件：** `gemm_v1_tiled_smem.cu`

线程协同将 A 和 B 的 16×16 tile 搬入共享内存，再从高速 SMEM 计算部分和。

- **核心思路：** 每个 tile 内的 A/B 元素仅从全局内存读取一次，在 SMEM 中复用 16 次
- **数据复用比：** 相比朴素实现提升 16×
- **预期收益：** 显著降低全局内存流量
Nsight可以看截图gemm_v1_tiled_smem.png，相较于Naive版本，warp stall的top3已经变成：
  a. stall MIO throttle: MIO 是 Memory Input/Output 单元的缩写，负责处理一些非内存读写但需要访问特殊硬件资源的操作。本质是：MIO/LSU 发射通道被内存类指令（尤其 shared/global 读写）压满了，warp 想再发指令但发不出去。
  b. stall barrier: 表示warp因为执行了同步指令（如 __syncthreads() 或 bar.sync）而被阻塞，正在等待同一线程块（CTA）内的其他线程也到达该同步点
  c. stall not selected: warp就绪但未被调度
原本较多的stall LG Throttle，从Naive版本的21.53%降低到0.1%，Stall Long Scoreboard从8.42%降低到2.97%。v1版本主要的问题是：stall MIO throttle 和 stall barrier。造成stall MIO throttle较大的原因是：从SMEM中读取频率较大。解决的策略主要是：提高数据在warp中的复用，摊薄SMEM的访存成本。

### V2：线程级分块（每线程 4×4）

**文件：** `gemm_v2_thread_tiling.cu`

每个线程计算 C 的一个 4×4 子块，并将部分和保存在寄存器中。

- **核心思路：** 每次共享内存读取可触发 16 次 FMA，线程级算术强度大幅提升
- **寄存器压力：** 每线程使用 16 个累加寄存器（速度快，无额外内存访问）
- **线程规模：** 每个 block 为 8×8 = 64 线程（对比 16×16 = 256），但每线程工作量提高 16×  
Nsight上最明显的改变是compute的SM占比从90%左右，下降到40%。可以看截图gemm_v2_thread_tiling.png，相较于v1版本，warp stall的top3已经变成：  
  a. Stall Long Scoreboard  
  b. Stall Short Scoreboard  
  c. stall LG Throttle从v1版本的20.43%降低到2.91%；  
可以得出，当前主要的性能阻塞点在Stall Long Scoreboard和Stall Short Scoreboard。Stall Long Scoreboard常见原因如下：  
  （i）tile 装载阶段的 global load 延迟没被完全隐藏  
  （ii）占用率/活跃 warp 不够（寄存器压力上来后更明显），导致“没别的 warp 可切换”  
  （iii）每轮 __syncthreads() 后很快消费刚加载数据，依赖链较紧  
Stall Short Scoreboard的常见原因如下：  
  （i）shared load 后立刻被 FMA 使用（读后用依赖）  
  （ii）累加器 sum[i][j] 连续更新，存在寄存器写后读链  
  (iii)unroll 后指令更密，调度器更容易碰到“下一条还没就绪”  
查看Nsight，发现ooccupacy都是75%。是寄存器限制了资源么？  
在Nsight的Impact of Varying Register Count Per Thread中，可以看到，当前每个线程平均使用48个寄存器，这是occupancy下降的主要原因。那么调整寄存器的使用是解决目前问题的好方案。
但是这里尝试了很多降低寄存器使用的方案，都几乎无效。因此，将优化的希望，放在ptx文件上，可以看当前生成的 build/artifacts/gemm/gemm_v2_thread_tiling_current.ptx。主要存在以下几个问题：
    a. 全局加载是标量 ld.global.nc.f32，指令数偏多
    b. 写回阶段有大量谓词分支（16 个元素分别判断）
先尝试向量化加载。

### V3：FastPath（4x8 + float4 写回）

**文件：** `gemm_v3_fastpath.cu`（`GemmV3KernelFast`）

在 V2 的基础上，V3 默认使用 `TMxTN=4x8` 线程级分块，并新增 fast path：

- **向量化加载：** A/B tile 主路径使用 `float4` 协同加载，减少标量 `ld.global` 指令数量
- **向量化写回：** C 写回使用 `float4` 向量 store，降低标量写回开销
- **主路径去边界分支：** 对齐尺寸（`M%32==0 && N%64==0 && K%8==0`）走无边界判断内核
- **双路径调度：** 对齐尺寸走 `GemmV3KernelFast`，非对齐尺寸自动回退 `GemmV3Kernel`
- **收益方向：** 降低 LSU 压力与谓词分支密度，提升大尺寸吞吐

Nsight 图片 v3.png 显示 compute SM 占比很高，warp stall 重新变为 stall MIO throttle 最高，说明共享内存带宽成为瓶颈。因此将 4×4 的线程级分块提升为 4×8 或 8×8。经测试 4×8 性能更好，设为 V3 默认。

### V3-P0/P1：大 Tile + 大 K-Tile 优化

**文件：** `gemm_v3_fastpath.cu`（`GemmV3LargeTileFast`）

在 V3 FastPath 基础上做两步优化：

**P0 — 增大 CTA Tile（32×64 → 128×128）：**
- 线程块从 8×8=64 增大到 **16×16=256**，每线程计算 8×8=64 个输出元素
- 计算/搬运比从 21 FMA/float 提升到 **64 FMA/float**（3x），大幅降低全局内存带宽需求
- 自适应调度：`M≥512 && N≥512` 且 128 对齐时走大 tile 路径，否则回退 32×64 路径
- **`__launch_bounds__(256, 2)`：** 提示编译器每 SM 至少保持 2 个 block

**P1 — 增大 K-Tile（8 → 32）：**
- `__syncthreads()` 同步频率从 K/8 降到 K/32（4x），减少 barrier stall
- 共享内存从 `As[128][9]+Bs[128][9]`≈9KB 增大到 `As[128][33]+Bs[128][33]`≈33KB，仍在 48KB 限制内
- 每次 K-tile 内计算量为 128×128×32 = 524,288 FMA，足够填满计算管线
- A/B 加载使用 float4 循环（128×32/4 = 1024 float4，256 线程各负责 4 次加载）

**kTileK 对比实测（4096×4096）：** K=8 → 10.0 TFLOPS | K=16 → 10.6 TFLOPS | **K=32 → 12.0 TFLOPS**

### V4：Tensor Core（WMMA）

**文件：** `benchmark_all.cu`（GemmTensorCoreKernel）

使用 NVIDIA WMMA（Warp Matrix Multiply-Accumulate）API 调用硬件加速矩阵乘。

- `nvcuda::wmma::mma_sync` 可在 Tensor Core 上执行矩阵乘加
- 当前实现采用 `16x16x8`（TF32 输入，FP32 累加）片段
- 每个 warp 负责一个 `16x16` 输出 tile，按 tile 维度映射 grid
- 对齐尺寸（16 的倍数）下可稳定运行，适合大尺寸 GEMM 吞吐优化
- **注意：** V4 直接从全局内存加载 fragment，无共享内存 tiling，大矩阵下带宽利用率不如 V3-P0



### cuBLAS 参考实现

**文件：** `gemm_cublas_ref.cu`

`cublasSgemm` —— NVIDIA 生产级 GEMM 实现，是本项目的性能目标。

## 性能结果

测试环境：Win11 + RTX 5060 Ti 16GB，CUDA 工具链 13.2（`sm_120`），数据来自 `build/bin/Release/data/results/*.csv`。


| M    | N    | K    | V0 Naive | V1 SMEM | V2 ThreadTile | V3 FastPath | V3-P0/P1 LargeTile | V4 TensorCore | cuBLAS  | 最快 / cuBLAS  |
| ---- | ---- | ---- | -------- | ------- | ------------- | ----------- | ------------------ | ------------- | ------- | -------------- |
| 128  | 128  | 128  | 0.011    | 0.014   | 0.020         | 0.020       | —（回退 V3）        | 0.015         | 0.018   | V0 / 0.63x     |
| 256  | 256  | 256  | 0.033    | 0.025   | 0.035         | 0.033       | —（回退 V3）        | 0.013         | 0.022   | V4 / 0.62x     |
| 512  | 512  | 512  | 0.187    | 0.136   | 0.091         | 0.079       | **0.068**          | 0.054         | 0.040   | cuBLAS / 1.00x |
| 1024 | 1024 | 1024 | 1.372    | 1.011   | 0.553         | 0.357       | **0.204**          | 0.278         | 0.170   | cuBLAS / 1.00x |
| 4096 | 4096 | 4096 | SKIP     | SKIP    | 33.395        | 19.820      | **11.455**         | 18.337        | 8.087   | cuBLAS / 1.00x |

结论：

- **V3-P0/P1（128×128 大 Tile + K-Tile=32）是当前最快的纯 CUDA Core 实现**：
  - `1024`：**0.204 ms / 10.5 TFLOPS**，相比原 V3 加速 **1.75x**，vs cuBLAS 仅 **1.20x**
  - `4096`：**11.455 ms / 12.0 TFLOPS（FP32 峰值 48%）**，相比原 V3 加速 **1.73x**，vs cuBLAS 仅 **1.42x**
- V3-P0/P1 在 512 以下自动回退到 V3 的 32×64 路径（小矩阵 block 数不够填满 SM）
- V4 TensorCore 在 1024/4096 反而慢于 V3-P0/P1，因为 V4 缺少共享内存 tiling，每次 K 迭代都直接访问全局内存
- cuBLAS 的 `cublasSgemm` 在 Ampere+ 架构上默认使用 TF32 Tensor Core，硬件峰值约为纯 FP32 的 2x，因此纯 FP32 路径的理论天花板约为 cuBLAS 的 50%
- 在 `128~1024` 全部 `PASS`，`4096` 为 `SKIP`（仅跳过 CPU 对拍）
- 为避免 Win11 图形驱动超时（WDDM TDR），V0/V1 对 4096 尺寸启用了 `SKIP_GPU_LARGE` 保护

## NVIDIA 参考 API

- **cuBLAS：** `cublasSgemm` / `cublasGemmEx`
- **cuBLASLt：** `cublasLtMatmul`（更灵活，支持自动调优）
- **CUTLASS：** 基于模板的高性能 GEMM 库

## 本次产物路径

- 可执行文件：`build/bin/Release/*.exe`（含 `gemm_v0~v3`、`gemm_cublas_ref`、`gemm_benchmark_all`）
- PTX/CUBIN/SASS：`build/artifacts/gemm/`
- 运行日志：`build/artifacts/gemm/runtime_logs/`
- 性能 CSV：`build/bin/Release/data/results/`

## V2 PTX/SASS 观察

基于 `build/artifacts/gemm/gemm_v2_thread_tiling.ptx` 与 `build/artifacts/gemm/gemm_v2_thread_tiling.sass`：

- **共享内存布局**：PTX 中 `As` 与 `Bs` 都是 1024 bytes（`8 × 32 × 4`），与代码里的 `TILE_K=8`、`BLOCK_ROWS=BLOCK_COLS=32`一致。
- **访存路径**：全局加载主要是 `ld.global.nc.f32`（对应源码 `__ldg`），SASS 中可见 `LDG.E.CONSTANT`；tile 写入共享内存对应 `st.shared.b32` / `STS`。
- **同步模式**：主循环每个 `k` tile 前后都有 `bar.sync`（SASS 中可见 `BAR`），保证共享内存读写有序。
- **计算密度**：核心计算段是大量 `fma.rn.f32`（SASS 对应 `FFMA/HFMA2`），符合线程级 4×4 累加器复用策略。
- **边界保护**：输出写回前使用大量 `setp` 谓词判断边界，再执行 `st.global.b32`，可覆盖非整齐尺寸。

