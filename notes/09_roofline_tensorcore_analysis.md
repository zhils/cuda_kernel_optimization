# Roofline 分析 + Tensor Core 性能评估

> 本文档回答：用不用 Tensor Core 的 kernel，怎么用 roofline 分析性能？为什么困惑？

---

## 1. 你的困惑本质

你困惑的核心是：**Tensor Core 让 roofline 变得复杂了——因为硬件不再是单一的 FP32 峰值线，而是多条线。**

```
传统 Roofline（FP32 CUDA Core only）：
  Performance
    ^
    |        ______memory bound______|__compute bound__
    |       /                       |
    |      /                        |  ———— FP32 peak (25 TFLOPS)
    |     /                         |
    |    /                          |
    |   /                           |
    |  /                            |
    | /                             |
    |/                              |
    +-----------------------------------> Arithmetic Intensity
    |          ↑                      ↑
  带宽斜率    你的 RMSNorm        你的 GEMM 4096³
  (448 GB/s)  (0.33 FLOP/Byte)   (683 FLOP/Byte)

Tensor Core Roofline（新增多条天花板线）：
  Performance
    ^
    |       |————— Tensor Core FP16 peak (51 TFLOPS with sparsity)
    |       |————— Tensor Core FP16 peak (25+ TFLOPS, no sparsity)
    |       |————— FP32 CUDA Core peak (25 TFLOPS)
    |      /
    |     /  ← 带宽斜率不变，但"撞天花板"的位置变了
    |    /
    +-----------------------------------> Arithmetic Intensity
```

**关键洞察**：Tensor Core 的峰值更高，但**只在足够高的算术强度下才能体现**。你要回答的核心问题是：**我的 kernel 的算术强度，在当前硬件上，Tensor Core 能带来多少收益？**

---

## 2. Roofline 的四条线（以 RTX 5060 Ti 为例）

```
线 1: 内存带宽天花板
  Performance_max = Arithmetic_Intensity × Memory_Bandwidth
  = AI × 448 GB/s
  
线 2: FP32 CUDA Core 天花板
  Performance_max = 25 TFLOPS （不随 AI 变化）

线 3: Tensor Core FP16 天花板（无结构化稀疏）
  Performance_max = 25+ TFLOPS （Blackwell 的具体数值取决于 FP16 accumulate throughput）

线 4: Tensor Core FP16 天花板（2:4 结构化稀疏）
  Performance_max = 51 TFLOPS
  （注意：结构化稀疏需要特殊的权重排列，你当前的 kernel 没有用到）

转折点计算（从 memory-bound → compute-bound 的 AI 阈值）：
  
  线 1→线 2: AI₁ = 25 TFLOPS / 448 GB/s = 56 FLOP/Byte  ← FP32 转折点
  线 1→线 3: AI₂ = 25+ TFLOPS / 448 GB/s ≈ 56+ FLOP/Byte ← Tensor Core 转折点（类似）
  线 1→线 4: AI₃ = 51 TFLOPS / 448 GB/s = 114 FLOP/Byte  ← 稀疏转折点
```

**关键发现：Tensor Core FP16 的转折点和 FP32 很接近（都是 ~56 FLOP/Byte），因为 FP16 峰值和 FP32 峰值在 RTX 5060 Ti 上相近。这意味着：Tensor Core 在"带宽瓶颈区"的辅助作用有限——它主要帮 compute-bound 区。但在数据中心 GPU（H100）上，FP16 峰值是 FP32 的 3×，转折点高了 3×，所以情况不同。**

---

## 3. 实际分析：你的每个 kernel 在 roofline 中的位置

### 3.1 算术强度计算

```
RMSNorm:
  计算量 = 4 × rows × cols （平方1 + 除法1 + 乘法2 per element）
  访存量 = 3 × rows × cols × 4B （读x + 读gamma + 写y）
  AI = 1/3 ≈ 0.33 FLOP/Byte

GEMM N×N×N:
  计算量 = 2 × N³
  访存量 = 3 × N² × 4B
  AI = N / 6 FLOP/Byte

Softmax:
  计算量 ≈ 3 × rows × cols（max + exp + normalize）
  访存量 ≈ 2 × rows × cols × 4B（读 + 写）
  AI ≈ 3/2 × (1/4) ≈ 0.38 FLOP/Byte（非常低！）
```

### 3.2 在 roofline 上的位置

| Kernel | 规模 | AI | 瓶颈区 | 离 FP32 天花板有多远 |
|--------|------|-----|--------|---------------------|
| RMSNorm | 任意 | 0.33 | **memory-bound** | 只需带宽的 0.33× |
| Softmax | 任意 | 0.38 | **memory-bound** | 只需带宽的 0.38× |
| GEMM | 128 | 21 | 偏 memory-bound | 21/56 = 38% of peak |
| GEMM | 256 | 43 | 过渡区 | 43/56 = 77% of peak |
| GEMM | 512 | 85 | **compute-bound** | 85 > 56 → 可撞天花板 |
| GEMM | 1024 | 171 | **compute-bound** | 171 >> 56 |
| GEMM | 4096 | 683 | **compute-bound** | 683 >> 56 |

### 3.3 Tensor Core 是否有利？逐规模分析

```
GEMM 128³ (AI=21 < 56):
  → memory-bound 区域
  → Tensor Core 更高峰值 → 无用（因为被带宽限制）
  → 实测：cuBLAS FP32 (0.024ms) 反而比 你的 v1 (0.0065ms) 慢
    原因：cuBLAS launch overhead + 小规模下 Tensor Core 的 setup cost
  → 结论：128³ 上，不用 Tensor Core 更好！

GEMM 256³ (AI=43 < 56):
  → 仍然偏 memory-bound
  → 你的 FP32 v3 (2.0 TFLOPS) ≈ cuBLAS (2.2 TFLOPS)
  → 差异不大

GEMM 512³ (AI=85 > 56):
  → compute-bound 区域开始！
  → 你的 FP32 v3: 9.1 TFLOPS (36% FP32 peak)
  → cuBLAS FP32:  8.8 TFLOPS (35%)
  → 在这个规模，FP32 还未饱和 → Tensor Core 提升空间有限
  → 理论最大（如果完美优化）：~16 TFLOPS（但受 occupancy 限制）

GEMM 1024³ (AI=171 >> 56):
  → 强 compute-bound
  → 你的 v3: 12.0 TFLOPS (48%)
  → cuBLAS: 13.9 TFLOPS (56%)
  → Tensor Core 的优势开始体现！

GEMM 4096³ (AI=683 >> 56):
  → 极强的 compute-bound
  → 你的 v3: 13.5 TFLOPS (54%)
  → cuBLAS: 17.1 TFLOPS (68%)
  → 差距扩大到 21% → Tensor Core 的优势在此规模最大
```

**核心结论：**

```
Tensor Core 的优势 随算术强度增大而增大：

  AI < 30:   Tensor Core 和 CUDA Core 区别不大（都受限于带宽）
  AI 30-100: 差异开始出现（10-20%）
  AI 100+:   Tensor Core 显著优于 CUDA Core（20-50%+）

所以：
  你的 RMSNorm/Softmax（AI < 1）：不需要 Tensor Core
  你的小 GEMM（128³, AI=21）：不需要 Tensor Core
  你的大 GEMM（4096³, AI=683）：Tensor Core 是关键路径！
```

---

## 4. 实际分析流程（手把手）

### Step 1: 获取你的 GPU 参数

```bash
# 获取理论带宽
nvidia-smi --query-gpu=memory.total,memory.free --format=csv
# 查产品 spec: RTX 5060 Ti DRAM bandwidth ≈ 448 GB/s

# 获取理论峰值
# 查 CUDA Programming Guide, 或运行 deviceQuery
./extras/demo_suite/deviceQuery
# 或在线查: https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units
```

### Step 2: 计算你的 kernel 的 AI 和实测性能

```python
# 计算工具
def arithmetic_intensity(flops, bytes_read, bytes_write):
    return flops / (bytes_read + bytes_write)

def achieved_tflops(ops, time_ms):
    return ops / (time_ms * 1e6)

# RMSNorm
flops = 4 * rows * cols
bytes_total = 3 * rows * cols * 4
ai = flops / bytes_total  # 0.33
achieved = achieved_tflops(flops, 0.3476)  # your V3 on 4096²
print(f"RMSNorm: AI={ai:.2f}, Achieved={achieved:.1f} TFLOPS")

# GEMM 4096³
flops = 2 * 4096**3
bytes_total = 3 * 4096**2 * 4
ai = flops / bytes_total  # 683
achieved = achieved_tflops(flops, 10.17)
print(f"GEMM 4096³: AI={ai:.0f}, Achieved={achieved:.1f} TFLOPS")
```

### Step 3: 画 roofline 图（或脑中画）

```
在图上标出你的 kernel 的 (AI, Performance) 点：

  Performance
  25 TFLOPS ┤                    ● ← GEMM 4096³ (理想)
            │              ● ← GEMM 4096³ (实测, 13.5 TFLOPS)
            │          ● ← GEMM 512³ (实测, 9.1 TFLOPS)
            │      ● ← GEMM 128³ (实测, 0.65 TFLOPS)
            │
     0.6 ───┼── ● ← RMSNorm (AI=0.33, 0.58 TFLOPS 带宽利用率)
            │ /
            │/
            +────────────────────────────────────> AI
            0.33    21      56    85     683
                    ↑             ↑
              memory-bound    compute-bound 分界线
```

### Step 4: 分析"距离天花板有多远"

```
距离 = 1 - (实测性能 / roofline天花板在该AI处的值)

GEMM 4096³:
  Roofline天花板 = min(25 TFLOPS, 683 × 448 GB/s) = 25 TFLOPS
  实测 = 13.5 TFLOPS
  距离 = 1 - 13.5/25 = 46%
  
  这 46% 去哪了？
    1. Occupancy 25% → 硬件利用率受限 → ~30% 损失
    2. WMMA API overhead → ~10% 损失  
    3. Bank conflict + instruction issue inefficiency → ~6% 损失

RMSNorm:
  Roofline天花板 = 0.33 × 448 GB/s = 147.8 GB/s = 0.58 TFLOPS
  实测 = 峰值带宽利用率 ~90% (已接近硬件极限)
  
  这个很好！
```

---

## 5. 面试时如何讲 Roofline

**面试官**："你怎么分析 kernel 性能？会不会用 roofline？"

**建议回答：**

> "会。我分三步：
>
> 第一步，定位瓶颈类型。先算算术强度 AI——GEMM 方阵是 N/6 FLOP/Byte，
> 我的 RTX 5060 Ti 转折点约 56 FLOP/Byte，所以 N<336 是 memory-bound，
> N>336 是 compute-bound。
>
> 第二步，判断 Tensor Core 是否需要。Tensor Core 的峰值更高，但只在
> compute-bound 区域有效。比如我的 RMSNorm AI=0.33，加 Tensor Core
> 毫无意义——它永远撞不到计算天花板，因为带宽才是瓶颈。而 4096³ GEMM
> AI=683，Tensor Core 的 25+ TFLOPS 峰值是必争之地。
>
> 第三步，分析实测点离天花板多远。我的 GEMM v3 在 4096³ 上 13.5 TFLOPS，
> 离 25 TFLOPS 天花板差 46%。我用 roofline 逆推这 46% 的构成：25%
> occupancy → ~30% 损失，WMMA API copy 开销 → ~10%，bank conflict →
> ~6%。这个分解帮我确定了下一步优化优先级——提升 occupancy 是第一要务。"
>

---

## 6. Tensor Core Roofline 速查表

| 你的问题 | 答案 |
|----------|------|
| Tensor Core 对小矩阵有用吗？ | **通常没用**：AI 太低，被带宽卡住；launch overhead 反而更大 |
| 怎么判断一个算子该不该用 Tensor Core？ | 算 AI → 看是否超过 FLOP/Byte 阈值 → 超过则建议用 |
| Tensor Core kernel 怎么分析性能？ | 用同一个 roofline，但天花板换成 Tensor Core 的峰值 |
| fp16 Tensor Core vs fp32 CUDA Core 怎么选？ | 算 AI 和两个峰值 → 看哪个更接近天花板 → 选能撞到的那个 |
| 为什么我的 WMMA kernel 性能不如 cuBLAS？ | WMMA 有 extra copy 开销 → 用 PTX mma.sync 可减少 |
| 带宽瓶颈区再加 Tensor Core 会怎样？ | 可能反而更慢（Tensor Core 的 setup 有固定开销） |
