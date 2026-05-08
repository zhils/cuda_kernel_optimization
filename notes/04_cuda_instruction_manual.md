# CUDA 关键指令学习手册

> 本文档回答：`__dp4a` 是什么？还有哪些指令必须学？按照学习优先级排序。

---

## 1. 指令分类速查

| 类别 | 指令 | 作用 | 学习优先级 | 你用过吗 |
|------|------|------|-----------|---------|
| **INT8 点积** | `__dp4a` | INT8 4元素点积+累加 | 🔴 P0 | ❌ |
| **Tensor Core** | `mma.sync.aligned` | FP16/FP8/INT8 矩阵乘加 | 🔴 P0 | 部分（WMMA） |
| **只读缓存** | `__ldg` | 通过只读缓存加载 | 🔴 P0 | ✅ |
| **流式加载** | `__ldcs` | 绕过 L1 的流式加载 | 🟡 P1 | 部分 |
| **异步拷贝** | `cp.async` | 异步 global→shared | 🟡 P1 | ✅ |
| **Prefetch** | `prefetch` | 预取 global→L1 | 🟢 P2 | ❌ |
| **Warp Shuffle** | `__shfl_down_sync` | warp 内数据交换 | 🔴 P0 | ✅ |
| **Warp Vote** | `__all_sync` | warp 内全投票 | 🟢 P2 | ❌ |
| **Fast Math** | `__expf`, `__sinf`, `rsqrtf` | 快但精度降的数学函数 | 🟡 P1 | 部分 |
| **FMA** | `__fmaf_rn` | 单次融合乘加 | 🟢 P2 | ✅（隐式） |
| **Atomic** | `atomicAdd` | 原子操作 | 🟢 P2 | 部分 |
| **PTX Assembly** | `asm volatile(...)` | 直接写 PTX 指令 | 🔴 P0 | ❌ |

---

## 2. `__dp4a` — INT8 四元素点积

### 2.1 是什么？

```cpp
// __dp4a 是一条 INT8 精度的 SIMD 指令
// 一次执行：4 个 INT8 元素两两点乘，结果加到 INT32 累加器中

int __dp4a(int a, int b, int c);
//            ^^^^  ^^^^  ^^^^
//            4×i8  4×i8  int32累加器

// 数学表示：
// c += a.byte0 * b.byte0 + a.byte1 * b.byte1 + a.byte2 * b.byte2 + a.byte3 * b.byte3
```

### 2.2 完整的量化推理 GEMM kernel Demo

```cpp
#include <cuda_runtime.h>
#include <cstdint>

// INT8 量化 GEMM kernel（使用 __dp4a）
// 计算：C = A_int8 × B_int8 + bias
//       A: [M, K], B: [K, N], C: [M, N]
//       A_scale, B_scale: 量化缩放因子（FP32）

__global__ void Int8GemmDp4aKernel(
    const int8_t* __restrict__ A,    // [M, K], INT8
    const int8_t* __restrict__ B,    // [K, N], INT8
    const float* __restrict__ A_scale,  // per-channel/row scale
    const float* __restrict__ B_scale,
    const float* __restrict__ bias,     // [N]
    float* __restrict__ C,              // [M, N], FP32 output
    int M, int N, int K) {

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    // INT32 累加器（比 FP32 累加更大范围，防止溢出）
    int acc = 0;

    // 每次处理 4 个 K 元素（因为 __dp4a 处理 4×i8）
    for (int k = 0; k < K; k += 4) {
        // 打包 4 个 INT8 到一个 int32
        // 注意：需要正确处理字节序和边界
        int packed_a = 0;
        int packed_b = 0;

        // 从 INT8 数组打包到 int32
        // little-endian: byte0 = LSB
        if (k + 3 < K) {
            // 全部 4 个元素都存在
            packed_a = *reinterpret_cast<const int*>(&A[row * K + k]);
            packed_b = *reinterpret_cast<const int*>(&B[k * N + col]);  // 注意 B 的步长！
            // B 的列读取不是连续的——这是 INT8 GEMM 的一个性能问题
            // 优化方案：提前对 B 做列→行 transform，或使用 shared memory tile
        } else {
            // 边界处理：不够 4 个时填充 0
            for (int kk = 0; kk < 4 && k + kk < K; kk++) {
                reinterpret_cast<int8_t*>(&packed_a)[kk] = A[row * K + k + kk];
                reinterpret_cast<int8_t*>(&packed_b)[kk] = B[(k + kk) * N + col];
            }
        }

        // —— 关键指令：一条 dp4a = 4 次乘加 ——
        acc = __dp4a(packed_a, packed_b, acc);
    }

    // 反量化：acc_int32 → float，应用 scale
    float result = static_cast<float>(acc) * A_scale[row] * B_scale[col];

    // Bias
    if (bias != nullptr) result += bias[col];

    C[row * N + col] = result;
}
```

### 2.3 进阶：共享内存 + dp4a 高性能版本

上面的 naive 版本有很大的性能问题：B 的列读取不是连续的。需要共享内存来优化。

```cpp
constexpr int kTM = 8;   // 每线程计算 8 行
constexpr int kTN = 8;   // 每线程计算 8 列
constexpr int kTileK = 32;  // 注意：需要是 4 的倍数
constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kWarpM = 16;
constexpr int kWarpN = 16;

__global__ void Int8GemmOptimizedKernel(
    const int8_t* __restrict__ A_packed,  // [M, K/4] 或 [M, K]（已打包）
    const int8_t* __restrict__ B_packed,  // B 建议提前转置为 [N, K] 以连续读取
    float* __restrict__ C,
    int M, int N, int K) {

    // 共享内存：INT8 tile（比 FP32 tile 小 4×！可以放更多的数据）
    __shared__ int8_t As[kBlockM][kTileK];
    __shared__ int8_t Bs[kTileK][kBlockN];

    int acc[kTM][kTN] = {0};  // INT32 累加器

    // 加载 tile → __syncthreads → 计算（用 __dp4a）→ 下一 tile
    // ... 结构同 FP32 GEMM，但加载时直接读 INT8 而不是 float4

    // dp4a 内循环（处理 K 维度，每次 4 个）
    for (int k = 0; k < kTileK; k += 4) {
        for (int tm = 0; tm < kTM; tm++) {
            // 打包 A 行的 4 个 INT8
            int packed_a = *reinterpret_cast<const int*>(&As[thread_row * kTM + tm][k]);

            for (int tn = 0; tn < kTN; tn++) {
                // 打包 B 列的 4 个 INT8
                int packed_b = *reinterpret_cast<const int*>(&Bs[k][thread_col * kTN + tn]);

                acc[tm][tn] = __dp4a(packed_a, packed_b, acc[tm][tn]);
            }
        }
    }

    // 反量化 + 写回 FP32
    for (int tm = 0; tm < kTM; tm++) {
        for (int tn = 0; tn < kTN; tn++) {
            int row = block_row + thread_row * kTM + tm;
            int col = block_col + thread_col * kTN + tn;
            if (row < M && col < N) {
                C[row * N + col] = static_cast<float>(acc[tm][tn]) * row_scale[row] * col_scale[col];
            }
        }
    }
}
```

### 2.4 INT8 vs FP32 性能对比

| 指标 | FP32 GEMM | INT8 GEMM + dp4a | 提升 |
|------|-----------|-----------------|------|
| 输入大小 | 16B / 4 elements | **4B / 4 elements** | **4×** 访存减少 |
| 计算吞吐 (理论) | N/6 FLOP/Byte | ~N/6 FLOP/Byte | 相当，但访存少 |
| 共享内存用量 | 128×32×4B=16KB | **64×32×1B=2KB** | **8×** SMEM 减少 |
| 寄存器压力 | FP32 acc | INT32 acc | 相同（32bit） |
| 反量化开销 | 无 | 1 mul/result | ~3-5% 额外开销 |

**核心收益：更小的 INT8 tile → 可以在共享内存中放更大的 tile（更多复用）或提高 SM 并行度（更多 block）。**

---

## 3. Tensor Core：WMMA vs PTX mma.sync

### 3.1 两种 API 对比

```cpp
// ====== 方式 1：WMMA API（简单，但有 extra copy 开销） ======
#include <mma.h>
using namespace nvcuda;

// 声明 fragment
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

// 加载 → 计算 → 存储
wmma::load_matrix_sync(a_frag, A_half + offset, lda);
wmma::load_matrix_sync(b_frag, B_half + offset, ldb);
wmma::fill_fragment(c_frag, 0.0f);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(C_half + offset, c_frag, ldc, wmma::mem_row_major);

// 问题：fragment 是寄存器的抽象，编译后可能有额外的 copy 指令


// ====== 方式 2：PTX mma.sync（直接控制，性能更好） ======
// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//     {d0, d1, d2, d3},    ← 4 个 FP32 累加器
//     {a0, a1, a2, a3},    ← 4 个 FP16 A 寄存器（2 个 FP16 打包/寄存器）
//     {b0, b1},             ← 2 个 FP16 B 寄存器
//     {c0, c1, c2, c3};    ← 4 个 FP32 C 输入

// 具体形状取决于 mma 指令的 M,N,K 参数
// 可以用内联汇编直接调用：
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
    : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
    : "r"(a_packed0), "r"(a_packed1), "r"(a_packed2), "r"(a_packed3),
      "r"(b_packed0), "r"(b_packed1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3));

// 优点：
// 1. 零中间 copy——数据直接走寄存器→Tensor Core→寄存器
// 2. 可以精细控制数据排布（不依赖 WMMA fragment layout）
// 3. CUTLASS 和 cuBLAS 内部用的就是这个

// 缺点（学习曲线）：
// 1. 需要理解 PTX 寄存器布局
// 2. 不同架构的 mma 指令格式不同（sm70 vs sm80 vs sm90）
// 3. 需要手动管理 FP16↔FP32 打包
```

### 3.2 WMMA 的性能开销实测（来自你的 GEMM V4）

```
原因分析：你的 V4 在 4096³ 上只有 77% 的相对 cuBLAS 性能
核心瓶颈：WMMA API 的 fragment 抽象 → 编译后有 extra copy 指令
         + 仅 25% occupancy → SM 利用率不足

```

---

## 4. `__ldg` vs `__ldcs` vs 普通加载

```cpp
// ====== 三种全局内存加载方式 ======

// 1. 普通加载：通过 L1 + L2 cache
float val = A[idx];
// 路径：Global → L2 → L1 → Register
// 适用：数据有复用（如 GEMM tile 的数据会在 K 循环中被多次读取）

// 2. __ldg：通过只读缓存（texture/L1-RO cache）
float val = __ldg(&A[idx]);
// 路径：Global → L2 → Read-Only Cache → Register
// 优点：不占用 L1 写缓存空间；对只读数据，命中率可能更高
// 适用：输入数据（A, B, weight），不会在 kernel 内修改

// 3. __ldcs：Streaming 加载（绕过 L1，只走 L2）
float val = __ldcs(&A[idx]);
// 路径：Global → L2 → Register（BYPASS L1）
// 优点：释放 L1 给 SMEM；对一次性读取的数据更好
// 适用：GEMV 类操作（每个数据只读一次，不需要 L1 缓存）

// ====== 选择决策 ======
//                        数据有复用    数据只用一次
//   只读（WMMA load）     __ldg         __ldcs
//   读写（输出）          普通          普通
//   流式（streaming）     __ldcs        __ldcs
```

---

## 5. Warp Shuffle 家族

```cpp
// ====== __shfl_down_sync：向下广播 ======
// Warp lane[0] 的值给 lane[0], lane[1] 的值给 lane[1], ... 
// lane[i] 获得 lane[i + offset] 的值

__device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;  // 所有 lane 都有完整的归约结果（在 lane 0）
}
// 可视化（8 元素 warp 为例）：
// lane:     0    1    2    3    4    5    6    7
// offset=4:
// shfl(4):  4    5    6    7    4    5    6    7  
// val 汇总: 0+4  1+5  2+6  3+7  4    5    6    7
// offset=2:
// shfl(2):  2+6  3+7  2+6  3+7  ...
// val 汇总: 0+4+2+6  ...
// offset=1:
// shfl(1):  1+5+3+7  ...
// 最后 lane[0] = sum_all

// ====== __shfl_xor_sync：蝶形交换 ======
// lane[i] 获得 lane[i ^ mask] 的值
val += __shfl_xor_sync(0xFFFFFFFF, val, 1);
val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
// 比 shfl_down 稍快（因为可以并行）

// ====== __shfl_sync：广播 ======
// 把 lane[srcLane] 的值广播给所有 lane
float broadcast_val = __shfl_sync(0xFFFFFFFF, val, 0);  // 广播 lane 0

// ====== __all_sync：warp 内投票 ======
// 检查 warp 内所有活跃线程的条件
if (__all_sync(0xFFFFFFFF, val > 0)) {
    // warp 内所有线程的 val > 0 时才进入
}
```

---

## 6. Fast Math 函数

```cpp
// ====== 标准 vs Fast Math ======
// 精度差异：通常在 1e-3 到 1e-4 量级
// 性能差异：通常 1.5-2x

float val;

// 标准版本                          Fast Math 版本
val = expf(val);          // →    val = __expf(val);        ~2x faster
val = 1.0f / sqrtf(val);  // →    val = rsqrtf(val);        ~2x faster
val = sinf(val);          // →    val = __sinf(val);        ~1.5x
val = cosf(val);          // →    val = __cosf(val);        ~1.5x
val = logf(val);          // →    val = __logf(val);        ~2x
val = powf(a, b);         // →    val = __powf(a, b);      ~1.5x

// ====== 什么时候用 ======
// ✅ 推理（inference）场景 — 精度损失可接受
// ✅ 激活函数（SiLU, GELU）— 1e-3 误差对模型结果几乎无影响
// ✅ Normalization 的 sqrt — rsqrtf 在 RMSNorm 中很常见
// ❌ 训练（training）的反向传播 — 累积误差可能导致梯度不稳定
// ❌ 精确的数值计算（如 loss 的计算）
```

---

## 7. PTX Inline Assembly 入门

```cpp
// ====== 什么是 PTX inline assembly ======
// PTX 是 CUDA 的中间指令集（介于 C++ 和 GPU 机器码之间）
// 用 asm volatile 可以直接在 CUDA kernel 中嵌入 PTX 指令

// 基本语法：
// asm volatile(
//     "ptx_instruction.format %0, %1, %2;"   // PTX 指令
//     : "=r"(output)                          // 输出操作数
//     : "r"(input1), "r"(input2)             // 输入操作数
//     : "memory");                             // clobber list

// ====== Demo 1：cp.async（异步拷贝） ======
asm volatile(
    "cp.async.ca.shared.global [%0], [%1], %2, %3;"
    :: "r"(static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr))),
       "l"(static_cast<unsigned>(__cvta_generic_to_shared(global_ptr))),
       "n"(16),       // 拷贝 16 字节
       "r"(is_last)); // 是否最后一组（commit group）

// ====== Demo 2：cp.async.commit_group + wait_group ======
asm volatile("cp.async.commit_group;");     // 提交一组
asm volatile("cp.async.wait_group 0;");     // 等待最近一组完成

// ====== Demo 3：ldmatrix（Tensor Core 加载指令） ======
// 从共享内存加载 16x16 tile 到寄存器
uint32_t regs[4];  // 4 个 32bit 寄存器
asm volatile(
    "ldmatrix.sync.aligned.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
    : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
    : "r"(static_cast<unsigned>(__cvta_generic_to_shared(smem_addr))));
// 之后 regs[0..3] 可以直接传给 mma.sync 做矩阵乘加
```

---

## 8. 学习优先级路线图

```
P0（基础）：
  ✅ __shfl_down_sync — 你已经在用了
  ⬜ __dp4a — 完成上面的 INT8 GEMM demo
  ⬜ mma.sync.aligned — 读一遍 CUTLASS tiled_mma 的 warp-level 实现
  ⬜ asm volatile 基础 — 能在你的 kernel 里写两条 PTX 指令
  ⬜ __ldcs — 对比 __ldg 在 GEMV 上的表现

P1（本月掌握，晋升需要的深度）：
  ⬜ cp.async.commit_group/wait_group — 你已经用了 cp.async
  ⬜ __expf / rsqrtf — 在你的 RMSNorm kernel 里替换
  ⬜ ldmatrix（Tensor Core 配套加载指令）

P2（进阶储备，高级职位要求）：
  ⬜ __all_sync / __any_sync — warp 内协同决策
  ⬜ atomicAdd / atomicCAS — 多 block 协同需要
  ⬜ __fmaf_rn — 理解 FMA 对精度的影响
  ⬜ prefetch — Hopper+ 的数据预取
```

---

## 9. 学习建议

对每条指令，遵循以下学习路径：

```
1. 读文档（CUDA Programming Guide + PTX ISA）
2. 写一个 10 行的 demo kernel（只做这条指令的事）
3. 用 cuobjdump -s 看生成的 SASS（验证编译器是否正确使用硬件指令）
4. 在你的 RMSNorm/GEMM 里替换一个调用
5. 测试正确性和性能变化
```

> `__dp4a` 是 INT8 的 SIMD 内积指令，一次做 4 个 INT8 的乘加。
