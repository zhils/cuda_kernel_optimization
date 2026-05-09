# Softmax CUDA 优化

## 数学定义

```
softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

**算术强度：**
- 计算量 ≈ 3 × rows × cols FLOPs（max + exp + sum + normalize）
- 访存量 ≈ 2 × rows × cols × 4B（读 x + 写 y）
- 强度 ≈ 0.38 FLOP/Byte → **强 memory-bound**

---

## 版本演进

| 版本 | 做法 | Memory Throughput | Compute Throughput | Occupancy |
|:----|------|:-----------------:|:------------------:|:---------:|
| v0 | 朴素三遍扫描（max → exp/sum → normalize），每行 1 线程 | ~90% | ~15% | 高 |
| v1 | SMEM staging + float4 向量化 + warp 内归约 | 待测量 | 待测量 | 待测量 |
| v2 | 8-warp 在线归约，增大并行度 | 待测量 | 待测量 | 待测量 |
| v3 | **Online 单遍算法 + SMEM + Warp Shuffle** | **84.87%** | 13.46% | 36.17% |

性能数据来自 `ncu --set basic`（1024×1024 规模）。

### v0 — 朴素基线

每行一个线程串行处理：先遍历找 max，再遍历算 exp+sum，再遍历归一化。

- 全局内存读 2 遍，写 1 遍
- 无 block 内协作
- 正确性验证基线

### v1 — 共享内存 + float4 向量化

每个 block（4 warp）处理一行，引入共享内存缓存：

- 用 float4 向量化加载 x 到 SMEM，指令数降为 1/4
- SMEM 缓存 exp 中间结果，避免写回全局内存
- 串行归约求 max 和 sum

### v2 — Warp Shuffle 归约

在 v1 基础上将 SMEM 树归约替换为 `__shfl_down_sync` warp shuffle：

```
// warp shuffle 蝶形归约
float sum = x_part;
sum += __shfl_down_sync(0xffffffff, sum, 16);
sum += __shfl_down_sync(0xffffffff, sum, 8);
sum += __shfl_down_sync(0xffffffff, sum, 4);
sum += __shfl_down_sync(0xffffffff, sum, 2);
sum += __shfl_down_sync(0xffffffff, sum, 1);
```

消除 SMEM 读写 round-trip，减少 bar.sync 次数。

### v3 — Online 单遍算法 + Warp Shuffle

核心改进引入 **Online Softmax** 算法，将 max + sum 两遍归约合并为单遍：

```cuda
float local_max = -INFINITY;
float local_sum = 0.f;

for (int c = lane; c < cols; c += WARP_SIZE) {
    const float val = s_data[c];
    // Online 更新：老的 sum 需要用新的 max 调整
    if (val > local_max) {
        local_sum *= __expf(local_max - val);
        local_max = val;
    }
    local_sum += __expf(val - local_max);
}

// warp shuffle 归约（同时处理 max 和 sum）
for (int offset = 16; offset > 0; offset >>= 1) {
    float m2 = __shfl_down_sync(0xffffffff, local_max, offset);
    float s2 = __shfl_down_sync(0xffffffff, local_sum, offset);
    float new_max = fmaxf(local_max, m2);
    local_sum = local_sum * __expf(local_max - new_max)
              + s2 * __expf(m2 - new_max);
    local_max = new_max;
}
```

结合 float4 向量化加载和写回：

```cuda
const float4 d = *reinterpret_cast<const float4*>(s_data + c * 4);
float4 out;
out.x = __expf(d.x - row_max) * inv;
out.y = __expf(d.y - row_max) * inv;
out.z = __expf(d.z - row_max) * inv;
out.w = __expf(d.w - row_max) * inv;
*reinterpret_cast<float4*>(row_y + c * 4) = out;
```

---

## Nsight Compute 瓶颈分析（2026-05-09）

命令：`ncu --set basic --target-processes all --kernel-name-base demangled`。  
统计口径：每个目标取 Duration 最大的一次 launch。

| 目标 | Max Duration(us) | Compute(SM) | DRAM | Memory | Achieved Occupancy | Reg/Thr | 结论 |
|:-----|-----------------:|------------:|-----:|-------:|-------------------:|--------:|:-----|
| `softmax_v0` | 647.65 | 0.69% | 2.78% | 12.30% | 16.64% | 40 | 朴素版本并行与访存效率都低 |
| `softmax_v1` | 86.02 | 79.99% | 11.74% | 79.99% | 33.55% | 42 | 当前实现中性能最好，算存利用均高 |
| `softmax_v2` | 340.06 | 16.02% | 84.51% | 84.51% | 8.31% | 43 | 带宽受限明显 |
| `softmax_v3` | 338.94 | 12.96% | 84.86% | 84.86% | 8.31% | 40 | 与 v2 接近，瓶颈仍在 DRAM |
| `softmax_cudnn_ref` | 309.92 | 14.74% | 87.98% | 87.98% | 48.54% | 80 | 参考库路径同样偏带宽瓶颈 |
| `softmax_benchmark_all` (`NCU_QUICK=1`) | 623.65 | 0.17% | 1.42% | 3.37% | 17.36% | 39 | 聚合入口用于烟测，非单核优化依据 |

补充：
- `softmax_benchmark_all` 已支持 `NCU_QUICK=1`（缩小 case + 仅跑 v0~v3），用于避免 NCU 长时间超时。
- 原始报告位于 `data/ncu_reports/text/softmax_*.txt`。

---

## 关键 PTX 指令

| 操作 | PTX 指令 |
|------|----------|
| 向量化加载 | `ld.global.nc.v4.f32` |
| 快速指数 | `ex2.approx.ftz.f32` |
| Warp 归约 | `shfl.sync.down.b32` |
| 共享内存读写 | `st.shared.f32` / `ld.shared.f32` |

---

## 构建与运行

```bash
cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=120 && make softmax_v3 -j$(nproc)
cd ..
./build/bin/softmax_v3
```

运行 benchmark：

```bash
make softmax_cudnn_ref -j$(nproc)
./build/bin/softmax_cudnn_ref
```
