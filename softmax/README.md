# Softmax CUDA 优化复盘

---

## 1. 版本演进

| 版本 | 文件 | 说明 |
|:----|:-----|:-----|
| **v0** | `softmax_v0.cu` | 朴素基线：每线程独立加载一行，计算 exp+sum+归一化 |
| **v1** | `softmax_v1.cu` | 4-warp 共享内存：warp 内归约 + SMEM 中间值交换 |
| **v2** | `softmax_v2.cu` | 8-warp 在线归约：增大并行度 |
| **v3** | `softmax_v3.cu` | SMEM 共享 + warp shuffle 协同：最优版本 |

## 2. Nsight Compute 瓶颈分析

使用 `ncu --set basic` profiling（规模 1024×1024）：

| 版本 | Memory Throughput | Compute Throughput | Occupancy | 主要瓶颈 |
|:----|:-----------------:|:------------------:|:---------:|:---------|
| **v3** | **84.87%** | 13.46% | 36.17% | SMEM 占用限制 occupancy |
| **v0** | ~90% | ~15% | 高 | DRAM 带宽 + 指令数 |

**Softmax 的天然瓶颈：**
- **计算轻量**：expf + 归约 + 归一化，计算吞吐远低于 GEMM
- **访存敏感**：每行需要读一次+写一次，全局内存带宽是主要瓶颈
- **SMEM 版本 v3 的 tradeoff**：使用 48KB+ SMEM 共享中间结果，减少全局访问，但 occupancy 降至 36%

## 3. PTX / SASS

PTX 和 SASS 文件位于 `softmax/asm/` 下：

```bash
softmax/asm/ptx/softmax_v0.ptx
softmax/asm/ptx/softmax_v3.ptx
softmax/asm/sass/softmax_v0.cubin
softmax/asm/sass/softmax_v3.cubin
```

关键 PTX 指令：
- **expf**：`ex2.approx.ftz.f32` — 指数运算
- **shfl**：`shfl.sync.down.b32` — warp 内归约
- **st.shared / ld.shared**：SMEM 读写（v3 warp 协同）

## 4. 产物路径

- **可执行文件：** `build/bin/softmax_v0` … `softmax_v3`
- **结果 CSV：** `data/results/`
- **ncu 报告：** `build/data/ncu_reports/`
- **PTX/SASS：** `softmax/asm/ptx/`、`softmax/asm/sass/`
- **CUDA 架构：** RTX 5060 Ti，Compute Capability **sm_120**，CUDA 13.2
