# CUDA Kernel 优化策略选择指南

> 本文档回答：增大 tile、向量化、减少 bank conflict、异步拷贝——**什么时候该用什么策略？**

---

## 1. 首先搞清楚你的算子在哪个瓶颈区

优化策略的选择必须先回答一个问题：**你的算子是访存受限（memory-bound）还是计算受限（compute-bound）？**

### 1.1 判断公式：算术强度

```
算术强度 = 总 FLOPs / 总访存字节数

算术强度 > GPU 的 FLOP/Byte 比值 → compute-bound（计算受限）
算术强度 < GPU 的 FLOP/Byte 比值 → memory-bound（访存受限）
```

### 1.2 常见算子的分类

| 算子 | 算术强度 | 瓶颈类型 | 优化主线 |
|------|----------|----------|----------|
| Element-wise (add, mul, relu) | ~0.25 | **强 memory-bound** | 向量化、融合、减少访存次数 |
| RMSNorm / LayerNorm | ~0.33 | **强 memory-bound** | 向量化、warp 归约、weight 缓存 |
| Softmax | ~0.5-1 | **memory-bound** | 共享内存、online 算法 |
| GEMV (矩阵乘向量) | ~1-2 | **memory-bound** | 不用分块！直接 cuBLAS 更优 |
| GEMM 128x128 | ~21 | 偏 memory-bound | 小 tile 可能比大 tile 好 |
| GEMM 256x256 | ~43 | 偏 compute-bound | 大 tile + 寄存器分块 |
| GEMM 1024x1024 | ~170 | **强 compute-bound** | 大 tile、流水线、Tensor Core |
| Conv1D 小核 | ~10-50 | 偏 compute-bound | 大 tile、implicit gemm |
| Self-Attention (QK^T) | ~d_model | 取决于 d | 小 d：memory-bound；大 d：compute-bound |

### 1.3 以你的 RTX 5060 Ti 为例

- FP32 峰值：~25 TFLOPS
- 内存带宽：~448 GB/s
- **FLOP/Byte 阈值 = 25000 / 448 ≈ 56 FLOP/Byte**

算术强度 < 56 → memory-bound；> 56 → compute-bound。

---

## 2. 四类策略的决策树

```
                    ┌──────────────────┐
                    │  你的算子是哪种？  │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
      Memory-bound     中间地带       Compute-bound
      (A.I. < 10)     (10~100)       (A.I. > 100)
              │              │              │
    ┌─────────┼───┐    ┌────┼────┐    ┌────┼────┐
    ▼         ▼   ▼    ▼    ▼    ▼    ▼    ▼    ▼
  向量化   融合  减少  增大  向量  bank  增大  异步  bank
  优先   优先  访存  tile  化   conflict tile 拷贝  conflict
```

---

## 3. 策略一：增大 Tile Size

### 3.1 什么时候应该增大 tile？

**核心原则：只有当数据有复用时，增大 tile 才有收益。**

```
数据复用次数 R = 每个数据被使用的次数

计算量          2 * M * N * K
R = ───────── = ───────────────
    访存量        M*K + K*N

对于 GEMM：R ≈ 2MN / (M + N)   （方阵时 R ≈ N）
对于 Conv：R ≈ kernel_size * output_size / (kernel_size + output_elements_per_tile)
```

**判断标准：**

| 数据复用次数 R | 结论 |
|----------------|------|
| R = 1（如 GEMV、element-wise） | **不要增大 tile**，增大反而增加同步开销 |
| R = 2~8 | 小 tile（16-32）合适 |
| R = 8~32 | 中等 tile（64-128）合适 |
| R > 32 | 大 tile（128-256）合适，但要配合寄存器分块 |

### 3.2 来自本项目的真实教训

**GEMV 的 tile 灾难（q_path_fusion_v2 初版）：**

```cpp
// ❌ 错误示范：给 GEMV 上 128x128 tile
// 问题：每列数据在 GEMV 中只被用 1 次（R=1），
//      但 256 个线程需要 256 次 __syncthreads()
// 结果：比朴素 v1 慢了 12.7 倍！

// ✅ 正确做法：GEMV 直接用 cuBLAS，让 L2 cache 处理复用
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            cols, rows, cols, &alpha, dwq, cols, dnorm, cols, &beta, dq, cols);
```

**GEMM 的 tile 选择（gemm_v0 → v2）：**

```cpp
// v0: 无 tile（R ≈ N，但无共享内存复用）
// v1: 16x16 tile — 对小矩阵足够（R=16，tile 够用）
// v2: 128x128 tile + 8x8 寄存器子块 — 对大矩阵有效（R=128）

// 关键：不是 tile 越大越好，而是 tile 要匹配数据复用次数
// 128x128 tile 对 128x128 矩阵：R=128，刚好吃满
// 128x128 tile 对 256x256 矩阵：R=256，tile 显得偏小
```

### 3.3 增大 tile 的代价

每增大一次 tile：
1. 寄存器压力增大 → occupancy 下降
2. 共享内存占用增大 → 并发 block 减少
3. 需要配合寄存器分块（TM×TN）才能吃满算力

**决策 checklist：**
- [ ] 数据复用次数 R 是否显著大于 tile 中的线程数？
- [ ] 增大 tile 后 occupancy 是否仍然 ≥ 25%？
- [ ] 是否配合了寄存器分块（每线程算多个输出元素）？
- [ ] 共享内存 bank conflict 是否已处理？

---

## 4. 策略二：向量化（float2/float4/half2）

### 4.1 什么时候应该向量化？

**向量化本质上是用更少的指令处理更多的数据，减少指令开销。**

**判断条件（满足越多越好）：**

1. **内存访问是连续的**（行主序矩阵的行方向访问）
2. **访问步长是向量宽度的整数倍**
3. **起始地址已对齐**（float4 需要 16 字节对齐）
4. **数据量远大于向量宽度**（否则边缘处理开销大于收益）

### 4.2 什么时候不要向量化？

| 场景 | 原因 |
|------|------|
| 跨行读取（列方向访问） | 地址不连续，每次只取 1 个元素 |
| 起始地址未对齐 | 会产生两次内存事务 |
| 数据量太小（< 4× 向量宽度） | 标量已经够快 |
| Kernel 已经 compute-bound | 瓶颈在计算不在指令发射 |

### 4.3 Demo：RMSNorm 的向量化演进

```cpp
// v0: 标量访问 — 4 条 ld 指令/4 个元素
for (int c = 0; c < cols; c++) {
    float v = x[r * cols + c];
    sum += v * v;
}

// v1: float4 向量化 — 1 条 ld 指令/4 个元素
// 条件：cols 是 4 的倍数，起始地址对齐
float4 x4;
for (int c = tid * 4; c < cols; c += block_size * 4) {
    x4 = *reinterpret_cast<const float4*>(&x[r * cols + c]);
    sum += x4.x * x4.x + x4.y * x4.y + x4.z * x4.z + x4.w * x4.w;
}

// v3: 条件向量化 — 对齐走 float4，不对齐走标量
if (cols % 4 == 0) {
    // float4 路径：指令数减少 4 倍
    #pragma unroll
    for (int c = tid * 4; c + 3 < cols; c += num_threads * 4) {
        float4 v = reinterpret_cast<const float4*>(row_x)[c / 4];
        thread_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
} else {
    // 标量 fallback
    #pragma unroll
    for (int c = tid; c < cols; c += num_threads) {
        float v = row_x[c];
        thread_sum += v * v;
    }
}
```

### 4.4 性能收益估算

RMSNorm 4096列，无向量化 vs float4：

| 指标 | 标量 | float4 |
|------|------|--------|
| ld 指令数/行 | 4096 | 1024 |
| 指令发射开销 | 基准 | 1/4 |
| 预期加速比 | 1x | 1.5-2x（实测 v0→v3：MS 从 1.26ms → 0.35ms，加速 3.6x） |

> 注意：实际加速不只是向量化的功劳，还有 warp 归约、weight 缓存等协同。

---

## 5. 策略三：减少 Bank Conflict

### 5.1 什么是 bank conflict？

共享内存有 32 个 bank（每个 4 字节宽）。同一 warp 内不同线程访问同一 bank 的不同地址 → bank conflict → 串行化。

```
bank_size = 4 bytes × 32 banks = 128 bytes/row

举例：float[32][32] 共享内存
- bank[tid] = 地址[tid] / 4 % 32 = tid % 32
- 理想：每个 tid 访问不同 bank → 1 次事务
- 冲突：tid 0 和 tid 32 访问相同 bank → 2 次事务
```

### 5.2 什么时候需要处理 bank conflict？

**判断公式（快速版）：**

```cpp
// 如果你的共享内存是这样声明的：
__shared__ float smem[TILE_M][TILE_K];  // TILE_K 是 32 的倍数

// 访问模式是：
float val = smem[threadIdx.x][k];  // 所有线程访问同一行不同列

// 则：bank_id = k % 32
// 如果 k % 32 在所有线程中相同 → 严重的 bank conflict
```

**两阶段判断：**

| 阶段 | 检查内容 | 什么时候需要改 |
|------|----------|----------------|
| 加载阶段 | warp 内线程加载的列偏移是否 % 32 重复 | GPU 上的共享/常量/texture加载不产生 bank conflict（这些缓存在硬件处理的L1中） |
| 计算阶段 | warp 内线程读取共享内存列的偏移是否 % 32 重复 | **当你写 `smem[threadIdx.x][k]` 或 `smem[k][threadIdx.x]` 时** |

### 5.3 Padding 策略

**最简单有效的方法：给共享内存数组加 padding。**

```cpp
// ❌ 有 bank conflict 的声明
// 当 TILE_K = 32 时，bank(0) = 0, bank(32) = 0, bank(64) = 0 → 同 bank！
__shared__ float As[TILE_M][TILE_K];

// ✅ 加 4 个 padding（float 型）
//    bank(0) = 0, bank(32) = bank(32) % 32 = 0 → 仍然冲突！
//    需要加奇数个 padding
__shared__ float As[TILE_M][TILE_K + 1];

// ✅✅ 经典 padding：+4（4 bytes × 4 = 16 bytes，错开 4 个 bank）
//     bank(0)=0, bank(36)=(36*4/4)%32=36%32=4 → 不冲突！
__shared__ float As[TILE_M][TILE_K + 4];
```

### 5.4 为什么是 +4 而不是 +1？

```
+1: TILE_K=32 → 新宽度=33 → bank(33×thread_row + col) 
    当 thread_row=0: bank 不变 → 对 row=0 访问无效
    当 thread_row>0: bank 每行偏移 1 → 部分有效
    
+4: TILE_K=32 → 新宽度=36
    第 1 行：bank(0..31) = 0..31 ✓
    第 2 行：bank(36..67) = 4..35 = 4..31, 0..3 → 与第 1 行完全错开！
```

### 5.5 Demo：本项目 GEMM 的共享内存 padding

```cpp
// gemm_v2.cu — 处理 bank conflict
constexpr int kBlockM = 128;  // 16 threads × 8 = 128
constexpr int kBlockN = 128;  // 16 threads × 8 = 128
constexpr int kTileK = 16;

// 加 padding 防止两阶段 bank conflict：
//   阶段1（加载）：warp 内线程并行加载同行不同列 → 列偏移 % 32 不重复 → 无冲突
//   阶段2（计算）：`for k` 循环中所有线程读 `As[thread_row][k]` → k 对所有线程相同 → 无冲突
//   阶段2（计算）：读 `Bs[k][thread_col]` → k 对所有线程相同 → 
//     各线程访问 Bs 的不同列，bank(thread_col) = thread_col % 32 → 无冲突
//
// 结论：当 TILE_K=16 且以 tile_K 为内循环时，不需要 padding！
// 但如果你使用的是 128x128 的大 tile 且 TILE_K=128，就需要 +4 padding
__shared__ float As[kBlockM][kTileK + 4];
__shared__ float Bs[kTileK + 4][kBlockN];
```

### 5.6 实战：怎么判断我需不需要 padding？

```bash
# 使用 nvcc 编译时生成 bank conflict 报告
nvcc -Xptxas -v,-warn-lmem-usage,-warn-spills your_kernel.cu

# 或者用 nsight compute 查看
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared \
         l1tex__data_pipe_lsu_wavefronts_mem_shared \
    your_executable
```

两个指标的比值 = 平均 bank conflict 次数：
- `~1.0`：无冲突，不需要改
- `~2.0`：2-way conflict，可能需要改
- `> 4.0`：严重冲突，**必须改**

---

## 6. 策略四：异步拷贝（cp.async）

### 6.1 什么是 cp.async？

把全局内存→共享内存的数据搬运和计算**重叠**，让 SM 在等待下一个 tile 数据时不闲着。

```
同步拷贝（v1/v2）：
  时间: |--load A/B--||--compute--||--load A/B--||--compute--|
  
异步拷贝 + 双缓冲（v3/v4）：
  时间: |--load tile_0--||--load tile_1--||--load tile_2--|
         {             }{  compute tile_0 }  {  compute tile_1 }
                          ^^^^^^^^^^^^^^
                          load 和 compute 重叠了！
```

### 6.2 什么时候用 cp.async？

**三步判断法：**

```
1. 你的 kernel 有多个 K-tile 吗？
   ├─ No  → 不需要（只有 1 个 tile，重叠不了）
   └─ Yes → 进入第 2 步
   
2. K-tile 加载时间 ≥ 计算时间？
   ├─ Yes（偏 memory-bound） → 收益巨大，强烈建议
   └─ No  → 进入第 3 步
   
3. K-tile 数量 ≥ 4？
   ├─ No  → 前几次 tile 加载无重叠，收益有限
   └─ Yes → 推荐使用（最后几个 tile 也有重叠机会）
```

### 6.3 什么时候不需要 cp.async？

| 场景 | 原因 |
|------|------|
| K 维度很小（K < 256） | tile 数太少，双缓冲开销大于收益 |
| Compute-bound 严重 | load 很快就完成了，重叠窗口太小 |
| 小矩阵（M,N < 256） | 一个 warp 就处理完了，不需要复杂的流水线 |
| Shared memory 已经不够用了 | 双缓冲 ×2 SMEM，可能放不下 |

### 6.4 Demo：cp.async + 双缓冲实现

```cpp
// cp.async + 双缓冲的完整实现
#include <cuda/barrier>

#define CP_ASYNC(cg, dst, src, size) \
  asm volatile("cp.async.cg.shared.global [%0], [%1], %2;" \
               :: "r"(static_cast<unsigned>(__cvta_generic_to_shared(dst))), \
                  "l"(static_cast<unsigned>(__cvta_generic_to_shared(src))), \
                  "n"(size))

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kTileK = 32;

__global__ void GemmCpAsyncKernel(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {

    // 双缓冲共享内存 — 注意 padding
    __shared__ float As_buf0[kBlockM][kTileK + 4];
    __shared__ float Bs_buf0[kTileK + 4][kBlockN];
    __shared__ float As_buf1[kBlockM][kTileK + 4];
    __shared__ float Bs_buf1[kTileK + 4][kBlockN];

    // 寄存器累加器（每线程 8×4）
    float accum[8][4] = {};

    const int block_row = blockIdx.y * kBlockM;
    const int block_col = blockIdx.x * kBlockN;
    const int thread_row = threadIdx.y;  // 0..15
    const int thread_col = threadIdx.x;  // 0..15

    int k_tile_idx = 0;
    const int num_k_tiles = K / kTileK;

    // —— 第 0 轮：加载到 buf0（无计算重叠） ——
    const float* A0 = A + (block_row + thread_row) * K + k_tile_idx * kTileK;
    const float* B0 = B + (k_tile_idx * kTileK + thread_row) * N + block_col + thread_col;

    for (int k = 0; k < kTileK; k += 1) {
        if (thread_col < kTileK) {
            As_buf0[thread_row * 8 + k / 4][k % kTileK + 4] = A0[k * thread_col]; // simplified
        }
    }
    // 简化示意，实际用 float4 + cp.async

    k_tile_idx++;

    // —— 第 1..N-1 轮：加载到次缓冲，计算当前缓冲 ——
    for (; k_tile_idx < num_k_tiles; k_tile_idx++) {
        // 加载下一个 tile 到 buf1（cp.async）
        // ... (load to buf1)

        // 同时计算 buf0 中的数据
        #pragma unroll
        for (int k = 0; k < kTileK; k++) {
            for (int tm = 0; tm < 8; tm++) {
                int row_a = thread_row * 8 + tm;
                float a = As_buf0[row_a][k];
                for (int tn = 0; tn < 4; tn++) {
                    int col_b = thread_col * 8 + tn;
                    accum[tm][tn] += a * Bs_buf0[k][col_b];
                }
            }
        }

        // 交换缓冲
        // buf0 ↔ buf1
    }

    // —— 最后一轮：只计算，不加载 ——
    // ... compute buf0 ...

    // 写回
    for (int tm = 0; tm < 8; tm++) {
        for (int tn = 0; tn < 4; tn++) {
            int row = block_row + thread_row * 8 + tm;
            int col = block_col + thread_col * 4 + tn;
            if (row < M && col < N) C[row * N + col] = accum[tm][tn];
        }
    }
}
```

### 6.5 cp.async 性能对比（本项目实测）

| GEMM 规模 | v2（无 cp.async） | v3（cp.async + 双缓冲） | 提升 |
|-----------|-------------------|------------------------|------|
| 512³ | 4.29 TFLOPS | 9.12 TFLOPS | 2.1x |
| 1024³ | 9.91 TFLOPS | 12.03 TFLOPS | 1.2x |
| 4096³ | 8.24 TFLOPS | 13.52 TFLOPS | 1.64x |

> 512³ 提升最显著，因为此时 tile 加载时间占比最高（偏 memory-bound 区）。

---

## 7. 四策略综合决策矩阵

| 你的算子 | 第一步 | 第二步 | 第三步 | 第四步 |
|----------|--------|--------|--------|--------|
| Element-wise | **向量化**（float4） | 融合相邻算子 | N/A | N/A |
| RMSNorm | **向量化**（float4） | Warp 归约 | Weight 缓存 | N/A |
| Softmax | 共享内存 | Online 算法 | Warp 归约 | N/A |
| GEMV | **不用分块！** | 用 cuBLAS | N/A | N/A |
| GEMM 128³ | 小 tile (16-32) | 向量化加载 | **不用 cp.async** | N/A |
| GEMM 256³ | 中等 tile (64-128) | 寄存器分块 | bank conflict padding | cp.async (K≥4 tiles) |
| GEMM 512³ | tile=128 | 寄存器分块 8×4 | **bank conflict** | **cp.async** |
| GEMM 1024³ | tile=128-256 | 寄存器分块 8×8 | bank conflict | **cp.async** |
| GEMM 4096³ | tile=256 | 寄存器分块 | bank conflict | cp.async + **occupancy** |
| Conv1D | tile=128 | 向量化 | Implicit GEMM | 考虑 Tensor Core |
| Attention | 取决于 d | FlashAttention 思路 | 分块 + softmax online | N/A |

---

## 8. 面试应答模板

面试官问你："GEMM 优化，你怎么选择策略？"

**你的回答框架：**

1. **先分类**（30 秒）：
   "优化的第一步是判断算子的瓶颈类型。我通过算术强度来判断——GEMM 方阵的算术强度是 N/6 FLOP/Byte。在我用的 RTX 5060 Ti 上 FLOP/Byte 阈值约 56，所以 N<336 时偏 memory-bound，N>336 时偏 compute-bound。"

2. **说策略选择逻辑**（60 秒）：
   "对于 memory-bound 区域（小 N），策略优先级是减少访存 > 向量化 > 计算效率。所以我用较小的 16×16 tile、float4 向量加载、`__ldg` 只读缓存就够了，不需要 cp.async。
   
   对于 compute-bound 区域（大 N），策略升级为：增大 tile 到 128×128 以提升数据复用；每线程用 8×4 寄存器子块来隐藏延迟；共享内存加 +4 padding 消除 bank conflict；引入 cp.async + 双缓冲让数据搬运和计算重叠。"

3. **讲一个反例**（30 秒）：
   "我踩过一个重要的坑：在 Q_Path_Fusion 的 GEMV 操作上盲目用了 128×128 tile，结果比朴素版本慢了 12.7 倍。根本原因是 GEMV 每列数据只被用一次（零复用），但 256 个线程有 256 次 syncthreads 开销。改成 cuBLAS GEMM 后从 944ms 降到 8.66ms。这个案例教会我：**优化策略必须匹配数据复用模式，不是 tile 越大越好。**"

---

## 9. 关键要点速查

| 策略 | 什么时候用 | 什么时候不用 | 核心判断指标 |
|------|-----------|-------------|-------------|
| 增大 tile | 数据复用 R > tile_size/2 | R ≤ 1（GEMV/Element-wise） | 数据复用次数 R |
| 向量化 | 连续对齐访问 | 跨行/跨列访问 | 地址连续 & 对齐 |
| 减少 bank conflict | padding 后 occupancy 仍 OK | 本来就是 1-way | ncu metrics 查看冲突比 |
| 异步拷贝 | K-tile ≥ 4 且不是极强 compute-bound | K < 256 或 SMEM 不够 | tile 数 & 加载/计算比 |
