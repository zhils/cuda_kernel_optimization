# Softmax CUDA 优化复盘

## 1. 版本演进

| 版本 | 文件 | 做法 |
|:----|:-----|:-----|
| v0 | `softmax_v0.cu` | 每行单线程，显式两遍扫描（max → exp/sum → 归一化） |
| v1 | `softmax_v1.cu` | 4-warp SMEM：warp 内归约 + SMEM 中间值交换 |
| v2 | `softmax_v2.cu` | 8-warp 在线归约，增大并行度 |
| v3 | `softmax_v3.cu` | SMEM + warp shuffle 协同，最优版本 |

## 2. 数学定义

```
softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

朴素版本两遍扫描：第一遍找 max，第二遍算 exp+sum+归一化。
Online softmax 一遍扫描同时维护 running max 和 running sum。

## 3. Nsight Compute 瓶颈分析

`ncu --set basic`（1024×1024）：

| 版本 | Memory Throughput | Compute Throughput | Occupancy | 瓶颈 |
|:----|:-----------------:|:------------------:|:---------:|:-----|
| v3 | 84.87% | 13.46% | 36.17% | SMEM 占用限制 occupancy |
| v0 | ~90% | ~15% | 高 | DRAM 带宽 + 指令数 |

**Softmax 的天然瓶颈：**
- 计算轻量：expf + 归约 + 归一化，吞吐远低于 GEMM
- 访存敏感：每行读一次+写一次，全局内存带宽是主要瓶颈
- v3 的 SMEM tradeoff：用 48KB+ SMEM 共享中间结果，减少全局访问，但 occupancy 降到 36%

## 4. PTX/SASS

PTX 和 SASS 在 `softmax/asm/` 下。

关键 PTX 指令：
- 指数：`ex2.approx.ftz.f32`
- warp 归约：`shfl.sync.down.b32`
- SMEM 读写：`st.shared / ld.shared`（v3 warp 协同）

## 5. 产物路径

- 可执行文件：`build/bin/softmax_v0` … `softmax_v3`
- 结果 CSV：`data/results/`
- ncu 报告：`build/data/ncu_reports/`
- PTX/SASS：`softmax/asm/ptx/`、`softmax/asm/sass/`
