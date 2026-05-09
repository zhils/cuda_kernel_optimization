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

| Rows | Cols | V0 | V1 | V2 | V3 | CUB Ref | CPU |
|------|------|-----|-----|-----|-----|-----|---------|-----|
| 128 | 128 | 0.0281 | 0.0102 | 0.0070 | **0.0067** | 0.0042 | 0.0079 |
| 256 | 256 | 0.0846 | 0.0156 | 0.0175 | **0.0093** | 0.0071 | 0.0987 |
| 512 | 512 | 0.1645 | 0.0114 | 0.0147 | **0.0102** | 0.0087 | 0.1659 |
| 1024 | 1024 | 0.3265 | 0.0168 | 0.0123 | **0.0120** | 0.0166 | 0.7175 |
| 4096 | 4096 | 1.2624 | 0.3480 | 0.3528 | **0.3476** | 0.3487 | 13.5083 |

### 3.2 带宽（GB/s）

公式：`rows × cols × 4 × 2 / time`（读 x + 写 y，gamma 按 cols 大小不计入；之前误用 ×3 已修正）。

| Rows | Cols | V0 | V1 | V2 | V3 | CUB Ref |
|------|------|------|------|------|------|---------|
| 128 | 128 | 4.7 | 12.9 | 18.8 | **19.7** | 31.4 |
| 256 | 256 | 6.2 | 33.5 | 30.0 | **56.3** | 73.4 |
| 512 | 512 | 12.7 | 183.8 | 142.6 | **204.7** | 241.7 |
| 1024 | 1024 | 25.7 | 499.5 | 682.0 | **699.3** ⚡ | 505.9 |
| 4096 | 4096 | 106.3 | 385.7 | 380.5 | **386.2** | 384.9 |

> 1024² 带宽 > 448 GB/s：该规模数据仅 8 MB，全在 L2 cache（Blackwell 48MB）里，不需要走 DRAM。4096²（128 MB）才是真正的 DRAM 瓶颈。

### 3.3 相对 CUB 的耗时倍数

| Rows | Cols | V0/CUB | V1/CUB | V2/CUB | V3/CUB |
|------|------|--------|--------|--------|--------|
| 128 | 128 | 2.07x | 0.75x | 0.51x | **0.49x** |
| 4096 | 4096 | 2.44x | 0.67x | 0.68x | **0.67x** |

---

## 4. Nsight Compute 瓶颈分析

`ncu --set basic`（4096×4096）：

| 版本 | Memory Throughput | DRAM Throughput | Compute Throughput | Occupancy | 瓶颈 |
|:----|:-----------------:|:---------------:|:------------------:|:---------:|:-----|
| v1 | 51.45% | 38.02% | 51.45% | 72.82% | 均衡（occupancy 良好） |
| v3 | **85.30%** | **85.30%** | 6.00% | 52.71% | **DRAM 带宽饱和** |

v3 的 weight SMEM caching 把 DRAM 吞吐从 38% 拉到 85%，与上面 386/448=86% 的自洽。

---

## 5. PTX/SASS

PTX 和 SASS 在 `rmsnorm/asm/` 下。

关键 PTX 指令：
- 向量化加载：`ld.global.nc.v4.f32`
- warp 归约：`shfl.sync.down.b32`
- 快速倒数平方根：`rsqrt.approx.ftz.f32`

---

## 6. 产物路径

- 可执行文件：`build/bin/rmsnorm_v0` … `rmsnorm_v3`
- 设备代码唯一入口：`rmsnorm/rmsnorm_kernels.cuh`
- ncu 报告：`build/data/ncu_reports/`
- PTX/SASS：`rmsnorm/asm/ptx/`、`rmsnorm/asm/sass/`

## Nsight Compute 性能分析


使用 `ncu --set basic` 对每个可执行文件的第一个 kernel launch 进行 profiling。
运行环境：NVIDIA RTX 5060 Ti (Blackwell sm_120) | CUDA 13.2 | Nsight Compute 2026.1.1

| 版本 | Kernel | Duration(us) | Compute% | MemBW% | L1% | L2% | Occupancy% | Reg/Thread | Block | Grid |
|---|---|---|---|---|---|---|---|---|---|---|
| rmsnorm_v0 | RmsnormSerialKernel | 904.8 | 0.9% | 12.7% | 41.3% | 12.7% | 16.4% | 39 | 256 | 4 |
| rmsnorm_v1 | RmsnormSmemVecKernel | 144.1 | 50.0% | 50.0% | 53.1% | 9.4% | 8.3% | 40 | 128 | 250 |
| rmsnorm_v2 | RmsnormShuffleKernel | 58.9 | 11.3% | 62.8% | 21.6% | 22.5% | 8.3% | 40 | 128 | 250 |
| rmsnorm_v3 | RmsnormWeightSmemKernel | 2.9 | 1.9% | 6.5% | 2.9% | 3.3% | 8.3% | 46 | 128 | 32 |
| rmsnorm_cub_ref | RmsnormCubRefKernel | 2.9 | 8.1% | 7.5% | 16.6% | 2.8% | 44.2% | 36 | 256 | 128 |
**说明：** ncu `--set basic` 默认对程序的**第一个 kernel launch** 进行 profiling。对于 GEMM 等算子，这对应最小测试尺寸（128×128），GPU 远未饱和。因此表格中的 Compute% / MemBW% 表示的是**小尺寸下的资源利用率**，用于横向对比各版本的寄存器压力、occupancy 等结构性差异。大尺寸下的实际性能请参考各算子 README 中的完整 benchmark 表格。

