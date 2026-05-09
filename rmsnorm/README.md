# RMSNorm CUDA 优化复盘

## 1. 数学定义

```
y = x / sqrt(mean(x²) + eps) × gamma

mean(x²) = (1/C) × Σ_{i=0}^{C-1} xᵢ²
```

其中 x ∈ R^{R×C}，gamma ∈ R^C，eps = 1e-5（数值稳定常数）。

**算术强度：**
- 计算量 ≈ 4×R×C FLOPs（平方和 + rsqrt + 乘法缩放 + gamma 乘）
- 最小搬运 = R×C×4×2 bytes（读 x + 写 y，gamma 是 C 大小，可忽略）
- 强度 ≈ 4RC / 8RC = 0.5 FLOP/Byte → **典型 memory-bound**

RMSNorm 每个元素只做 O(1) 次计算，带宽决定速度。

---

## 2. 版本演进

| 版本 | 改造点 | 4096² 耗时 | 带宽 (GB/s) |
|:----|--------|:----------:|:-----------:|
| v0 | 每行单线程串行 | 1.2624 ms | 106 |
| v1 | SMEM staging + float4 | 0.3480 ms | 386 |
| v2 | + warp shuffle 归约 | 0.3528 ms | 381 |
| v3 | weight 缓存到 SMEM | 0.3476 ms | 386 |

v1/v2/v3 在 4096² 差距不大（~0.35ms），因为大尺寸下带宽已经触顶。小尺寸（128~1024）v3 优势明显。

### v0 — 每行单线程（基线）

```cuda
// 每行一个线程：串行算平方和 → rsqrt → 写回
for (int row = blockIdx.x; row < rows; row++) {
    float sum = 0;
    // 读一遍 x 算平方和
    for (int j = 0; j < cols; j++)
        sum += x[row * cols + j] * x[row * cols + j];
    float rms = rsqrt(sum / cols + eps);
    // 再读一遍 x 做归一化
    for (int j = 0; j < cols; j++)
        y[row * cols + j] = x[row * cols + j] * rms * gamma[j];
}
```
问题：x 读两遍，gamma 每行从全局内存读一遍，没有 block 内协作。**4096²: 1.2624 ms / 106 GB/s。**

### v1 — SMEM staging + float4 向量化

每行交给一个 block，SMEM 里做平方和归约，`float4` + `__ldg` 一次性读 4 个 float。

```
线程映射：grid(rows) × block(cols/4)
每个线程：从全局用 float4 读 x，存 SMEM
          归约 x² → rms
          从 SMEM 读回 x，做归一化，float4 写 y
```
gamma 还是每行从全局读——**这是 v3 要解决的事。**
**4096²: 0.3480 ms / 386 GB/s。**

### v2 — Warp shuffle 归约

v1 的 SMEM 归约（tree reduce）换成了 `__shfl_xor_sync` 寄存器归约：

```cuda
float sum = x_part * x_part;
// warp shuffle 树归约
sum += __shfl_xor_sync(0xffffffff, sum, 16);
sum += __shfl_xor_sync(0xffffffff, sum, 8);
sum += __shfl_xor_sync(0xffffffff, sum, 4);
sum += __shfl_xor_sync(0xffffffff, sum, 2);
sum += __shfl_xor_sync(0xffffffff, sum, 1);
// lane 0 得到完整的平方和
```

少了一次 SMEM 写回+读出的 round trip。**4096²: 0.3528 ms / 381 GB/s。**

### v3 — Weight 缓存到 SMEM + Warp 归约

gamma 一开始就从全局加载到动态 SMEM 并广播给所有 warp。

```
线程映射：block(128) = 4 warp，每个 warp 处理一行
SMEM 排布：
  gamma_smem[cols]  ← 启动时一次性加载 (cudaMemcpyToSymbol 风格)
                       每个线程协作搬 cols/128 个元素

归约路径：
  x² → warp shuffle sum → 存 SMEM → cross-warp SMEM 归约 → rms

读写路径：
  对齐时 float4，不对齐时 float
```

**4096²: 0.3476 ms / 386 GB/s。** 占理论带宽（448 GB/s）的 **86%**。

---

## 3. 性能数据

### 3.1 执行时间（ms）

| Rows | Cols | V0 | V1 | V2 | V3 | CUB | CPU |
|------|------|-----|-----|-----|-----|-----|-----|
| 128 | 128 | 0.0281 | 0.0102 | 0.0070 | **0.0067** | 0.0136 | 0.0079 |
| 256 | 256 | 0.0846 | 0.0156 | 0.0175 | **0.0093** | 0.0161 | 0.0987 |
| 512 | 512 | 0.1645 | 0.0114 | 0.0147 | **0.0102** | 0.0128 | 0.1659 |
| 1024 | 1024 | 0.3265 | 0.0168 | 0.0123 | **0.0120** | 0.0253 | 0.7175 |
| 4096 | 4096 | 1.2624 | 0.3480 | 0.3528 | **0.3476** | 0.5174 | 13.5083 |

### 3.2 带宽（GB/s）

公式：`rows × cols × 4 × 2 / time`（读 x + 写 y，gamma 按 cols 大小不计入；之前误用 ×3 已修正）。

| Rows | Cols | V0 | V1 | V2 | V3 | CUB |
|------|------|------|------|------|------|-----|
| 128 | 128 | 4.7 | 12.9 | 18.8 | **19.7** | 9.7 |
| 256 | 256 | 6.2 | 33.5 | 30.0 | **56.3** | 32.7 |
| 512 | 512 | 12.7 | 183.8 | 142.6 | **204.7** | 163.5 |
| 1024 | 1024 | 25.7 | 499.5 | 682.0 | **699.3** ⚡ | 331.6 |
| 4096 | 4096 | 106.3 | 385.7 | 380.5 | **386.2** | 259.4 |

> 1024² 带宽 > 448 GB/s：该规模数据仅 8 MB，全在 L2 cache（Blackwell 48MB）里，不需要走 DRAM。4096²（128 MB）才是真正的 DRAM 瓶颈。

### 3.3 相对 CUB 的耗时倍数

| Rows | Cols | V0/CUB | V1/CUB | V2/CUB | V3/CUB |
|------|------|--------|--------|--------|--------|
| 128 | 128 | 2.07x | 0.75x | 0.51x | **0.49x** |
| 4096 | 4096 | 2.44x | 0.67x | 0.68x | **0.67x** |

---

## 4. Nsight Compute 瓶颈分析（2026-05-09）

命令：`ncu --set basic --target-processes all --kernel-name-base demangled`。  
统计口径：每个版本取 Duration 最大的一次 launch。

| 版本 | Max Duration(us) | Compute(SM) | DRAM | Memory | Achieved Occupancy | Reg/Thr | 结论 |
|:-----|-----------------:|------------:|-----:|-------:|-------------------:|--------:|:-----|
| `rmsnorm_v0` | 908.22 | 0.95% | 4.55% | 12.81% | 16.48% | 39 | 基线版本利用率很低 |
| `rmsnorm_v1` | 697.54 | 51.31% | 39.07% | 51.31% | 8.33% | 40 | 算存均衡，但 occupancy 受共享内存限制 |
| `rmsnorm_v2` | 321.18 | 10.20% | 82.86% | 82.86% | 8.33% | 40 | 转为明显带宽受限 |
| `rmsnorm_v3` | 334.18 | 5.88% | 86.90% | 86.90% | 38.68% | 40 | DRAM 接近饱和，符合 memory-bound 预期 |

结论：v2/v3 的优化方向本质是把瓶颈“推”到内存带宽上；v3 的吞吐上限主要由 DRAM 决定。

---

## Warp Stall 原因分析

| 版本 | #1 Stall | #2 Stall | #3 Stall | #4 Stall | #5 Stall |
|:----|:---------|:---------|:---------|:---------|:---------|
| v0 | **Long Scoreboard 97.4%** | Wait 2.5% | No Instruction 0.0% | Not Selected 0.0% | Short Scoreboard 0.0% |
| v1 | Wait 34.7% | Long Scoreboard 31.0% | Short Scoreboard 21.3% | Mio Throttle 11.8% | No Instruction 1.2% |
| v2 | **Long Scoreboard 81.8%** | Wait 8.7% | Short Scoreboard 8.4% | Mio Throttle 0.7% | No Instruction 0.4% |
| v3 | **Long Scoreboard 78.7%** | Mio Throttle 11.0% | Short Scoreboard 8.8% | Wait 1.2% | Not Selected 0.2% |

结论：v0 和 v2/v3 均以 Long Scoreboard 为主导（>78%），说明这些版本在等待全局内存加载完成。v1 的 Wait 占 34.7%，反映其 SMEM 协作阶段线程同步占比更高。

---

## 5. PTX/SASS

PTX 和 SASS 可在本地通过 `cuobjdump -ptx <binary>` 或 `cuobjdump -sass <binary>` 生成（`**/asm/` 已从版本控制中排除）。

关键 PTX 指令：
- 向量化加载：`ld.global.nc.v4.f32`
- warp 归约：`shfl.sync.down.b32`
- 快速倒数平方根：`rsqrt.approx.ftz.f32`

---

## 6. 产物路径

- 可执行文件：`build/bin/rmsnorm_v0` … `rmsnorm_v3`
- ncu 报告：`data/ncu_reports/`
- PTX/SASS：本地运行 `cuobjdump -ptx <binary>` 生成
