# PTX/SASS 阅读与分析实战指南

> 本文档回答：怎么阅读 PTX？怎么分析？怎么编排学习计划？

---

## 1. 基础知识回顾（统一术语）

### 1.1 编译链条

```
CUDA C++ 源码 (.cu)
    │
    ▼  nvcc 前端
PTX (Parallel Thread Execution) — 虚拟指令集，与硬件无关
    │
    ▼  ptxas (PTX Assembler)
SASS (Shader Assembly) — 真实机器码，与具体 GPU 架构绑定
    │
    ▼  硬件执行
```

### 1.2 为什么需要读 PTX/SASS？

| 读什么 | 回答什么问题 |
|--------|-------------|
| **PTX** | 编译器是否正确展开了循环？是否用了 FMA 指令？Tensor Core 是否被调用？ |
| **SASS** | 寄存器是否 spilled 到 local memory？有多少 stall cycle？指令级延迟是如何隐藏的？ |
| **两者对比** | 为什么我的 WMMA kernel 比 PTX mma kernel 慢？SASS 里多了几条 copy 指令。 |

---

## 2. 如何生成 PTX/SASS

### 2.1 从源文件生成 PTX

```bash
# 方式 1：编译时保留中间文件
nvcc -ptx -o rmsnorm_v3.ptx rmsnorm_v3.cu

# 方式 2：编译时同时生成 PTX（保留中间文件）
nvcc -keep -o rmsnorm_v3 rmsnorm_v3.cu
# 生成 rmsnorm_v3.ptx（PTX 中间文件）

# 方式 3：从可执行文件中提取 PTX
cuobjdump -ptx ./build/bin/rmsnorm_v3 > rmsnorm_v3.ptx

# 方式 4：只看特定函数的 PTX
cuobjdump -fun RMSNormV3Kernel -ptx ./build/bin/rmsnorm_v3
```

### 2.2 生成 SASS

```bash
# 从可执行文件反汇编 SASS
cuobjdump -sass ./build/bin/gemm_v3 > gemm_v3.sass

# 带源码行号对照（编译器生成调试信息时）
nvcc -G -lineinfo gemm_v3.cu -o gemm_v3
cuobjdump -sass ./gemm_v3 > gemm_v3_with_debug.sass

# 只看特定函数
cuobjdump -fun GemmV3Kernel -sass ./build/bin/gemm_v3
```

### 2.3 一步到位脚本

```bash
#!/bin/bash
# gen_asm.sh — 为项目中的每个 kernel 生成 PTX 和 SASS
KERNEL=$1  # 例如 rmsnorm_v3

echo "=== PTX ===" > ${KERNEL}.asm
cuobjdump -ptx ./build/bin/${KERNEL} >> ${KERNEL}.asm
echo "" >> ${KERNEL}.asm
echo "=== SASS (SM120) ===" >> ${KERNEL}.asm
cuobjdump -sass ./build/bin/${KERNEL} >> ${KERNEL}.asm
echo "Done → ${KERNEL}.asm"
```

---

## 3. PTX 阅读实战

### 3.1 你的 RMSNorm kernel PTX（简化版）

生成一个 PTX 文件后，你会看到类似这样的内容：

```ptx
// ====== 寄存器声明 ======
.reg .f32   %f<100>;          // FP32 寄存器（最多 100 个）
.reg .pred  %p<10>;           // 谓词寄存器（条件判断）
.reg .u32   %r<50>;           // 无符号 32bit 寄存器（地址）

// ====== 数据移动 ======
ld.global.f32   %f1, [%rd1];     // 从全局内存加载 float
st.global.f32   [%rd2], %f3;     // 存储 float 到全局内存

// ====== 算术 ======
mul.f32     %f10, %f1, %f1;       // %f10 = %f1 * %f1  (平方)
add.f32     %f11, %f11, %f10;     // %f11 += %f10       (累加)
fma.rn.f32  %f12, %f1, %f2, %f3;  // %f12 = %f1*%f2 + %f3 (融合乘加)

// ====== 分支 ======
@%p0 bra BB_3;           // 如果 %p0 为真，跳到 BB_3

// ====== 共享内存 ======
st.shared.f32   [%r10], %f11;    // 存到共享内存
ld.shared.f32   %f12, [%r11];    // 从共享内存加载
bar.sync 0;                       // __syncthreads()

// ====== Warp Shuffle ======
shfl.sync.down.b32  %f20, %f11, 16, 31, -1;
// ↑ lane[i] 获得 lane[i+16] 的值，等价于 __shfl_down_sync

// ====== 控制流 ======
bra.uni BB_1;          // 无条件跳转
ret;                   // kernel 返回

// ====== Tensor Core (WMMA 展开后) ======
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%f100, %f101, %f102, %f103},
    {%r20, %r21, %r22, %r23},
    {%r24, %r25},
    {%f100, %f101, %f102, %f103};
// ↑ 1 条 mma 指令 = 16×8×16 维度的矩阵乘加 = 2048 次 FP16 FMA！
```

### 3.2 你的优化决策对应 PTX

| C++ 改动 | PTX 变化 |
|----------|----------|
| `float4 v = *reinterpret_cast<float4*>(ptr)` | 4 条 `ld.global.f32` → 1 条 `ld.global.v4.f32` |
| Warp shuffle 替换 SMEM 归约 | `st.shared→bar.sync→ld.shared` 链变成 `shfl.sync.down.b32` 链 |
| `__ldg(&ptr)` | `ld.global.nc.f32`（`.nc` = non-coherent，走只读缓存） |
| `__ldcs(&ptr)` | `ld.global.cs.f32`（`.cs` = cache streaming，绕过 L1） |
| `__syncthreads()` | `bar.sync 0;` |
| `__expf(val)` | `mul.f32` + `ex2.approx.f32`（快但近似） |
| `nvcuda::wmma::mma_sync(...)` | `mma.sync.aligned.m16n8k16...` 展开 |

### 3.3 分析 PTX 的三个核心问题

**打开你的 PTX 文件，找到你的 kernel 函数入口，逐一回答：**

```
问题 1：有没有 local memory spill？
  → 搜索 "st.local" 或 "ld.local"
  → 如果有 → 寄存器不够用了，数据溢出了 local memory（就是 HBM！慢！）
  → 解法：减小 tile size，或减少每线程的寄存器用量

问题 2：FMA 还是分立 mul+add？
  → 搜索 "fma.rn.f32" vs "mul.f32" + "add.f32"
  → fma = 1 条指令 1 次运算（但精度更高，因为只在最后舍入一次）
  → mul+add = 2 条指令（但有些场景编译器无法安全合并为 FMA）

问题 3：循环是否正确展开了？
  → 找到 K 循环 → 看有没有 bra 指令回到循环头
  → 展开的循环 = 整个循环体被复制 N 次，没有 bra
  → 未展开 = 有 bra，每次迭代都有跳转开销（~4-6 cycles）
```

---

## 4. SASS 阅读实战

### 4.1 SASS 长什么样（SM120 / Blackwell 为例）

```
// 你的 gemm_v3 SASS 的典型片段

/* 地址:           汇编:                                注解: */
/* 0x0000 */       IMAD.MOV R0, RZ, RZ, c[0x0][0x28] ; // 读取 blockIdx
/* 0x0010 */       CS2R.32 R8, SR_CLOCKLO ;              // 读取时钟
/* 0x0020 */       LDS.128 R12, [R10] ;                  // 从共享内存加载 128bit
/* 0x0030 */       FFMA R16, R12, R14, R16 ;             // FMA: R16 += R12*R14
/* 0x0040 */       @P0 BRA 0x20 ;                        // 谓词跳转（循环）
/* 0x0050 */       STG.128 [R18], R16 ;                   // 存储 128bit 到全局内存
/* 0x0060 */       EXIT ;                                // kernel 结束

// ====== 关键指令分类 ======
// LDS.128    — 从共享内存加载 16 字节（4 floats，= float4）
// STG.128    — 存储 16 字节到全局内存（也是 4 floats）
// FFMA       — Fused FMA（浮点融合乘加，Tensor Core 的 SASS 指令）
// IMAD       — Integer Multiply-Add（地址计算）
// CS2R       — 读特殊寄存器
// BRA        — 跳转
// @P0        — 条件执行（如果谓词寄存器 P0 为真）
```

### 4.2 SASS 回答优化问题

```
问题：为什么 V4 只有 25% occupancy？
  SASS 分析：
  1. cuobjdump -sass gemm_v4  → 查看 .section .info 段（如果有）
  2. 或直接数：用了多少寄存器？
     → 搜索所有 %r 寄存器，看最大编号
     → 如果 R255 → 意味着用了约 256 个 32bit 寄存器
     → SM120 每 SM 有 65536 个 32bit 寄存器
     → 256 threads × 256 regs = 65536 → 只能 1 个 block/SM！
     → occupancy = 100% × (1 block / max blocks_per_SM) = 12.5%

  解法：减少寄存器用量 → 重新设计 tile 参数（从 8x8 → 4x4 等）

问题：为什么 cp.async 在 V3 上 512³ 效果最好？
  SASS 分析：
  1. 看 STG（全局写入）和 FFMA（Tensor Core计算）之间是否有 stall
  2. 小 tile：STG 和 FFMA 间隔均匀 → 流水线跑满
  3. 大 tile：有连续的 STG 或连续的 FFMA → 等待，流水线断流
```

---

## 5. 实践：对比优化前后的 PTX/SASS

### 5.1 案例：RMSNorm V0 → V3

```bash
# 生成
cuobjdump -ptx ./build/bin/rmsnorm_v0 > rmsnorm_v0.ptx
cuobjdump -ptx ./build/bin/rmsnorm_v3 > rmsnorm_v3.ptx

# 比较
diff rmsnorm_v0.ptx rmsnorm_v3.ptx
```

**你应该发现的关键差异：**

| 对比维度 | V0 PTX | V3 PTX | 解释 |
|----------|--------|--------|------|
| 指令数 | ~200 行 | ~120 行 | V3 用 warp shuffle 替换了 SMEM 树归约 → 少了很多 bar.sync |
| 全局 load 数量 | 每行 2× ld | 每行 1× ld | V3 的 weight 被一次加载到 SMEM，后续复用 |
| 循环展开 | 无 | `#pragma unroll` → PTX 中循环体被复制 | 消除了循环跳转开销 |
| shared memory 声明 | 无 | `.shared .align 4 .b8 smem[...]` | V3 把 weight staging 到 SMEM |

### 5.2 案例：GEMM V2 (无 cp.async) vs V3 (cp.async)

```bash
cuobjdump -sass ./build/bin/gemm_v2 | grep -E "LDS|FFMA|STG|BAR" > v2.sass
cuobjdump -sass ./build/bin/gemm_v3 | grep -E "LDS|FFMA|STG|BAR" > v3.sass
```

**关键差异（SASS 层面）：**

```
V2 SASS pattern（同步拷贝）：
  LDS ...      ← load to shared     // 等
  BAR.SYNC     ← __syncthreads      // 等所有线程完成加载
  FFMA ...     ← compute            // 计算
  FFMA ...
  BAR.SYNC     ← __syncthreads      // 等计算完成，准备下一个 tile
  LDS ...      ← next load          // 又开始加载
  → load 和 compute 不能重叠 → BAR 之间有气泡

V3 SASS pattern（异步拷贝+双缓冲）：
  LDS ...      ← load tile_N          // 加载与计算并行
  CP.ASYNC ... ← async next load      // cp.async 启动下一批
  FFMA ...     ← compute tile_{N-1}   // 同时计算上一批
  FFMA ...
  CP.WAIT ...  ← wait for load        // 如果 load 没完成就等
  BAR.SYNC     ← 同步（仅当交换缓冲时）
  FFMA ...     ← compute tile_N
  → load 和 compute 有重叠窗口 → 流水线利用率更高
```

---

## 7. 7 天学习计划

```
Day 1：环境准备 + 生成第一个 PTX
  □ cuobjdump -ptx ./build/bin/rmsnorm_v0
  □ 找到你的函数入口（.entry RMSNormV0Kernel）
  □ 识别 5 种基本指令：ld.global、st.global、mul.f32、add.f32、bar.sync
  □ 对照 CUDA C++ 源码，找到 1-1 对应关系

Day 2：理解寄存器与数据流
  □ 数一数 kernel 用了多少个 %f 寄存器
  □ 把"从全局内存读 → 运算 → 写到共享内存"的指令链路画出来
  □ 找到 __syncthreads → bar.sync 0 的对应

Day 3：从 PTX 发现优化问题
  □ 搜索 "st.local"（如果有 = 寄存器溢出 → 需要减小 tile）
  □ 看 mma.sync 指令是否出现（你的 V4 应该有）
  □ 看有没有 FMA 还是分立 mul+add（大多数情况编译器会优化成 fma）

Day 4：生成并分析 SASS
  □ cuobjdump -sass ./build/bin/gemm_v3
  □ 找到 LDS、BAR.SYNC、FFMA、STG 这四个关键指令
  □ 画出它们在时间轴上的分布（人工读，或后面学 Nsight Compute 自动出图）

Day 5：优化前后 PTX/SASS 对比
  □ 拿 GEMM V0→V1→V2→V3 的 PTX，逐版本对比指令数的变化
  □ 找到具体从哪个版本开始出现 mma.sync 指令
  □ 总结一份表：每个版本的 PTX 行数、寄存器数、关键指令类型

Day 6：用 PTX/SASS 解释性能数据
  □ 为什么 V3 在 512³ 上快到 9.1 TFLOPS？
    → 从 SASS 中找到 LDS 和 FFMA 交替出现的证据（流水线跑满了）
  □ 为什么 V4 在 4096³ 只有 77% cuBLAS？
    → 搜索 st.local（寄存器溢出），看 FFMA 的密度（是否计算资源空闲）

Day 7：综合练习
  □ 修改你的 GEMM V3 的一个 tile 参数
  □ 重新生成 PTX/SASS，观察指令数的变化
  □ 总结：改了什么参数 → PTX/SASS 哪里变了 → 性能怎么变
  → 这个学习链是你面试时最有价值的展示！
```

---

## 8. PTX/SASS 分析速查表

| 你想知道什么 | PTX 中搜索 | SASS 中搜索 | 含义 |
|-------------|-----------|------------|------|
| 有没有 Tensor Core？ | `mma.sync` | `HMMA` 或 `IMMA` | Tensor Core 矩阵乘加 |
| 寄存器够不够？ | 最后一行 `%f<XX>` 看最大编号 | 出现 `STL`（store local） | XX>128 → 可能有压力 |
| cp.async 生效了？ | `cp.async` | `LDGDEPBAR` | 异步全局加载 |
| 循环展开了？ | 算K循环=1 or N | 没有 `BRA` 回到循环头 | K=1=展开；K>1=未展开 |
| 向量化做了？ | `ld.global.v4.f32` | `LDG.128` | float4 = 4×32bit=128bit |
| FMA 还是 mul+add？ | `fma` vs `mul`+`add` | `FFMA` | 1 条 vs 2 条指令 |
| 寄存器溢出了？ | `st.local` / `ld.local` | `STL` / `LDL` | local = HBM，非常慢 |
| 共享内存 bank conflict？ | 无法从 PTX 直接看 | 需要 Nsight Compute | SASS 只是指令，运行时的 bank 行为是硬件内部的事 |

---

## 9. 面试应答模板

面试官："你会读 PTX 和 SASS 吗？"

**回答（1-2 分钟）：**

> "我会。我的学习方法是：先保证能生成，再学会分析。
>
> 生成：我用 `cuobjdump -ptx` 和 `-sass` 从编译好的可执行文件提取
> PTX/SASS，并且写了脚本自动化这个过程——每个 kernel 版本都有对应的
> PTX/SASS 归档。
>
> 分析：我主要看三件事——
>
> 第一，Tensor Core 是否真的被调用了。在 PTX 里搜 `mma.sync`，确认
> 编译器正确生成了 Tensor Core 指令而不是 fallback 到 CUDA Core FMA。
>
> 第二，寄存器压力。看 PTX 里用了多少个寄存器，如果超过 128，可能
> occupancy 会掉到 25% 以下——我的 GEMM V4 就是这个问题。
>
> 第三，是否有 local memory spill。search `st.local`——如果有，说明
> 编译器把本该在寄存器里的数据踢到了 HBM，这是最严重的一类性能问题。
>
> 我还做了一个版本对比。拿 GEMM V2（无 cp.async）和 V3（有 cp.async）
> 的 SASS 对比——V2 的 LDS 和 FFMA 有明显的 BAR.SYNC 隔离，V3 中
> LDS 和 FFMA 交替出现，证明数据加载和计算成功重叠了。这个对比直接
> 解释了 V3 在 512³ 上 2.1x 加速的来源。"
