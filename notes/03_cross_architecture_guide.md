# 跨架构 GPU 适配指南

> 本文档回答：需不需要测试 AMD 显卡、华为昇腾、vGPU-48GB、L20 等其他 GPU？怎么适配？

---

## 1. 为什么需要关注跨架构？

在面试高性能算子开发工程师时，"跨架构"会以两种方式出现：

1. **浅层**：你了解不同 GPU 架构的差异吗？（考察知识广度）
2. **深层**：如果要把你的 kernel 移植到 AMD / 昇腾，你会怎么做？（考察工程思维）

**你的定位策略**：承认你主要在 NVIDIA 生态上实践过，但展示你对跨平台迁移的工程方法论有清晰认知。

---

## 2. NVIDIA GPU 架构世代对比

即便都是 NVIDIA，不同架构差异也非常大。先把你自己的 RTX 5060 Ti (Blackwell, sm_120) 放进去对比。

| 架构 | 代表 GPU | SM | Tensor Core | FP32 TFLOPS | 显存带宽 | 关键特性 |
|------|----------|-----|-------------|-------------|----------|----------|
| **Blackwell** (120) | RTX 5060 Ti | 新一代 | 第5代 | ~25 | ~448 GB/s | **你的卡**；FP4/FP6支持 |
| **Ada Lovelace** (89) | RTX 4090, L20 | 第3代 | 第4代 | ~83 | ~1008 GB/s | FP8 原生支持；Transformer Engine |
| **Hopper** (90) | H100, H800 | 第4代 | 第4代 | ~67 | ~2039 GB/s (HBM3) | TMA; FP8; DPX指令; Thread Block Cluster |
| **Ampere** (80) | A100, RTX 3090 | 第3代 | 第3代 | ~19.5 | ~1555 GB/s (HBM2e) | 结构化稀疏; MIG |
| **Turing** (75) | RTX 2080 Ti, T4 | 第2代 | 第2代 | ~13.4 | ~616 GB/s | 首次引入 Tensor Core |
| **Volta** (70) | V100 | 第1代 | **第1代** | ~15.7 | ~900 GB/s | 首个 Tensor Core |

### 2.1 对你代码的直接影响

```cpp
// 你的 GEMM kernel 用了这些特性：
//   __syncthreads()      → 所有架构都支持
//   float4               → 所有架构都支持
//   cp.async             → SM80+ (Ampere 起)
//   __ldg / __ldcs       → SM35+ (几乎都支持)
//   WMMA                 → SM70+ (Volta 起)
//   mma.sync.aligned     → SM70+ (但指令格式随架构变)
//   TMA                  → SM90+ (Hopper 起) — 你没有用
//   Thread Block Cluster → SM90+ (Hopper 起) — 你没有用

// 结论：你的 v1/v2 kernel 可运行在几乎所有 NVIDIA GPU 上
//       v3/v4 用了 cp.async → 只在 Ampere+ 上有效
```

### 2.2 同一 kernel 在不同架构上的行为差异

以你的 GEMM v3 为例：

```
RTX 5060 Ti (sm_120, 25 TFLOPS):
  512³ → 9.12 TFLOPS (36% 峰值利用率)
  
RTX 4090 (sm_89, 83 TFLOPS):
  512³ → ~20 TFLOPS (~24% 利用率，更多 SM 但 occupancy 可能更低)
  原因：SM 越多 → 每个 SM 分到的 tile 越少 → 空闲越多

H100 (sm_90, 67 TFLOPS):
  4096³ → ~40 TFLOPS (利用 TMA 减少 cp.async 开销)
```

---

## 3. AMD GPU 适配（ROCm/HIP）

### 3.1 核心差异（NVIDIA vs AMD）

| 特性 | NVIDIA CUDA | AMD ROCm/HIP | 影响 |
|------|-------------|-------------|------|
| 编程模型 | CUDA C++ | HIP C++（语法几乎一样） | 低：可以用 hipify 工具自动转换 |
| 线程束大小 | **Warp = 32** | **Wavefront = 64** | 🔴 **高：所有 warp-level 操作都要改** |
| 共享内存 | 可配置 L1/SMEM 比例 | 固定 | 低 |
| Tensor Core | WMMA/PTX mma | MFMA 指令 | 中：需要不同的 intrinsic |
| 编译工具 | nvcc | hipcc | 低 |
| 性能分析 | Nsight Compute/Systems | rocprof/rocprofiler | 中 |

### 3.2 最重要的差异：Warp 32 vs Wavefront 64

```cpp
// NVIDIA (warp = 32)
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// AMD (wavefront = 64) — 需要修改 offset 范围
__device__ float wavefrontReduceSum(float val) {
    for (int offset = 32; offset > 0; offset >>= 1)  // 从 32 开始！
        val += __shfl_down(val, offset);
    return val;
}

// 通用方案：用 warpSize 变量（CUDA = 32, HIP = 64）
__device__ float genericReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}
```

**对你代码的影响：**

你的 RMSNorm v3 用了 warp shuffle 归约，硬编码了 `warpSize=32`。移植到 AMD 需要：
1. 或用 `warpSize` 宏（HIP 自动设为 64）
2. 或重新设计 data partition（每 wavefront 处理的元素数量需要重新调参）

### 3.3 Tensor Core 等效：MFMA 指令

```cpp
// NVIDIA WMMA (FP16→FP32)
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, ...> a_frag;
nvcuda::wmma::fill_fragment(c_frag, 0.0f);
nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

// AMD MFMA (FP16→FP32)
// 使用不同的 intrinsic，形状也不同
// MI200 系列：MFMA_16x16x16_FP16_FP16_FP32
__builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
```

### 3.4 适配策略（不实际买 AMD 卡也能做）

```
1. 代码层面：
   - 把 `__shfl_down_sync(mask, ...)` 封装成 `WARP_SHUFFLE_DOWN(...)` 宏
   - 把 Tensor Core 调用封装成统一接口
   - 用 `#ifdef __HIPCC__` 区分编译路径

2. 验证策略（没钱买真的 AMD 卡）：
   - 用 hipify 把 CUDA 代码转成 HIP 代码
   - 在文档/README 中写明"理论上兼容，warp 处理已适配"
   - 面试时说："我设计了跨平台的 warp shuffle 抽象层，并分析了 warp/wavefront
     差异对 tile 参数的重新计算方法"
```

---

## 4. 华为昇腾（Ascend）适配

### 4.1 核心差异

| 特性 | NVIDIA CUDA | 华为 Ascend CANN | 影响 |
|------|-------------|-----------------|------|
| 编程语言 | CUDA C++ | **TIK C++ (Python DSL)** 或 Ascend C | 🔴 大：完全不同 |
| AI 计算单元 | Tensor Core (SM 内) | **Cube Unit** (独立) | 🟡 中 |
| 向量单元 | CUDA Core | **Vector Unit** (独立) | 🟡 中 |
| 内存层级 | HBM → L2 → SMEM → Reg | HBM → L2 → **L1 Buffer** → **UB (Unified Buffer)** | 🔴 大 |
| 线程模型 | 线程 → Warp → Block → Grid | 无传统线程模型 → 用 repeat/shift | 🔴 非常大 |

### 4.2 昇腾的独特编程模型

```cpp
// 在 NVIDIA 上，你会这样写 GEMM：
__global__ void gemmKernel(A, B, C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 线程自己算位置
    float sum = 0;
    for (int k = 0; k < K; k++) sum += A[row*K + k] * B[k*N + col];
    C[row * N + col] = sum;
}

// 在昇腾上（Ascend C），概念完全不同：
// 没有线程 → 有的是"指令"和"数据流"
// 数据搬迁：用 DataCopy 指令把数据从 HBM → UB
// 计算：用 Matmul（Cube Unit）或 Vec（Vector Unit）指令
// 类似于写汇编 + 手动管理数据流
```

### 4.3 对面试的影响

**你应该这样回答：**

> "昇腾的编程模型和 CUDA 完全不同——它用的是 Ascend C，不是 C++ 线程模型。
> 核心挑战在于数据流管理：UB(Unified Buffer) 的容量远小于 NVIDIA 共享内存，
> tile size 的选取策略完全不同。Cube Unit 和 Vector Unit 是独立的硬件单元，
> 需要你显式地编排它们的数据流——这和在 NVIDIA 上通过线程协作有本质区别。
> 
> 所以我如果要适配，会先学习 Ascend C 的 DataCopy + Matmul 双流水线模式，
> 然后重新设计 tile 大小和流水深度——而不是简单地移植代码。"

---

## 5. vGPU-48GB 与 L20

### 5.1 vGPU (Virtual GPU)

vGPU 是虚拟化环境中的 GPU，"48GB" 通常指 A40 或类似卡的 vGPU 切片。

| 特性 | 物理 GPU | vGPU |
|------|---------|------|
| 显存 | 完整 | **切片**（可能 6/12/24/48GB） |
| SM 数量 | 全部 | **切片**（可能 1/4, 1/2 等） |
| 驱动 | 标准 | **GRID/vGPU 驱动** |
| CUDA 支持 | 完整 | 必须用 vGPU 授权的驱动 |
| 开发调试 | 直接 Nsight | ⚠️ 可能受限（看 vGPU profile） |

**对你的关键影响：**

```cpp
// 在 vGPU 上，你的 block/grid 配置需要更保守
// 因为可用的 SM 可能只有物理卡的 1/4

// 你的 gemm_v3 在物理 RTX 5060 Ti 上：
//   SM = 34（假设），每个 SM 处理多个 tile
// vGPU (1/4 切片)：
//   SM = 8，同样的数据量 → 每个 SM 处理 4× 多的 tile → 寄存器可能不够

// 应对：vGPU 上应该减小 block size，增加 grid 并行度
```

### 5.2 L20 (NVIDIA L20)

L20 是 Ada Lovelace 架构 (sm_89)，48GB GDDR6，主要给推理场景。

```
L20 关键参数：
  - SM 数量：约 68
  - FP32 TFLOPS：~59.8
  - FP8 TFLOPS：~239（Transformer Engine）
  - 显存带宽：~864 GB/s
  - 功耗：275W（比 L40S 的 350W 低）
```

**对比你的 RTX 5060 Ti：**

| 参数 | RTX 5060 Ti | L20 |
|------|-------------|-----|
| SM 数 | ~34 | ~68 (2×) |
| FP32 TFLOPS | ~25 | ~59.8 (2.4×) |
| 带宽 | ~448 GB/s | ~864 GB/s (1.9×) |
| 显存 | 16GB? | **48GB (3×)** |
| FLOP/Byte | ~56 | ~69 |

**对你的 GEMM kernel：**

```
因为 L20 的 SM 数是你的两倍，同一个 kernel：
  - 中规模矩阵 (512²)：每个 SM 分到的 tile 更少 → 可能更慢
  - 大规模矩阵 (4096²)：更多的 SM 并行处理 → 明显更快
  
FLOP/Byte=69 vs 56 → L20 更偏 compute-bound → 更大的 tile 更有效
```

---

## 6. 你的面试策略

### 6.1 现状诚实陈述

> "我的所有实践经验都在 NVIDIA 生态上——CUDA C++。对 AMD ROCm/HIP 和华为
> Ascend CANN 我做过功能对比和迁移方案研究，但没有实际运行过。如果团队需要在
> 这些平台上部署，我可以快速上手，因为核心优化思想（分块、流水线、访存模式）
> 是相通的。"

### 6.2 展示的知识点（加分项）

| 知识点 | 怎么体现 |
|--------|----------|
| Warp(32) vs Wavefront(64) | 你代码中的 warp 归约使用了 `warpSize` 而不是硬编码 32 |
| 不同架构的 SM 数和带宽 | 在 GEMM 的 README 中写了不同规模在不同架构上的预期行为 |
| vGPU 的内存切片 | 了解 vGPU 的 SM 切片对 tile 大小的影响 |
| 不同卡的 FLOP/Byte 比 | 用 roofline 做不同卡的瓶颈分类 |

### 6.3 如果你真的需要测试其他 GPU

```
策略 1（低成本）：
  - AWS/阿里云 GPU 实例按需租用（AMD MI250X、NVIDIA L20）
  - 可以按小时计费，只在需要 benchmark 时租

策略 2（更低价）：
  - 关注社区 benchmark：MLPerf、Lamini 等
  - 对比官方数据后推演你的 kernel 表现

策略 3（理论推导，面试时可用）：
  - 用 roofline model：输入新 GPU 的峰值算力和带宽
  - 输出你的 kernel 在目标架构上的预期性能瓶颈
```

---

## 7. 快速参考卡

| 想做什么 | 需要学什么 | 关键差异点 |
|----------|-----------|-----------|
| NVIDIA→AMD 移植 | HIP + ROCm | **Wavefront=64** |
| 用 AMD Tensor Core | MFMA intrinsic | 指令形状不同 |
| 用 NVIDIA Hopper 特性 | TMA, DPX, Cluster | 只能在 H100/H800/B200 上跑 |
| 适配华为昇腾 | Ascend C (TIK) | **没有线程概念，数据流编程** |
| 适配 vGPU | 减小 block size | SM 切片，资源更少 |
| 适配 L20 | 增大 tile (FLOP/Byte 更高) | 更多 SM，更大显存 |
| 适配各类 GPU 的关键 | **Roofline knows all** | 输入峰值 FLOPS + 带宽 → 预测瓶颈 |
