# RMSNorm

## 数学定义

```
y = x / sqrt(mean(x²) + ε) × γ

其中 mean(x²) = (1/C) × Σ_{i=0}^{C-1} x_i²
```

RMSNorm（Root Mean Square Layer Normalization）在现代 LLM（如 LLaMA、Qwen 等）中被广泛使用。它是 LayerNorm 的更简洁替代：去掉均值中心化，仅用 RMS 做归一化。

## 理论性能分析

### 算术强度

| 指标 | 公式 | 数值 |
|------|------|------|
| 数据搬运量 | 读 x + 读 γ + 写 y ≈ (2R×C + RC) × 4 bytes | ~12RC bytes（量级） |
| 计算量 | R × (C 次乘加 + rsqrt + C 次乘加) | ~4RC FLOPs |
| 算术强度 | 约 4RC / 12RC | **~0.33 FLOP/Byte**（量级） |

### Roofline 分类

与 Softmax 类似，RMSNorm 也是典型 **内存受限** 算子。优化目标是最大化 DRAM 带宽利用率并减少全局读写次数。

### 理论峰值

- RTX 5060 Ti DRAM 带宽：~448 GB/s（量级）
- 理想下界（每元素读 x、读 γ、写 y 各一次）：约 3 × R × C × 4 bytes 搬运量（γ 按每行重复读 C 次计）

## 优化版本

### V0：朴素实现（每行单线程）

**文件：** `rmsnorm_v0_naive.cu`

每个线程处理整行：先串行计算 `sq_sum`，再串行写归一化结果。

- **问题：** 行内无并行，SM 利用率低
- **问题：** 每行对 `x` 需完整遍历两次（平方和与缩放）

### V1：减少全局访存 + 优化访存形式

**文件：** `rmsnorm_v1.cu`

相对 V0 仅做全局内存两类优化：

1. **减少全局读写：** V0 每行对 `x` 读两遍。当 `cols` 可在共享内存中暂存整行时，先把 `x` 读入共享内存一次，平方和与写 `y` 均从共享读，全局对 `x` 的读取减半；`cols` 过大时走两阶段 stream 路径，并用 **block 内树形归约** 汇总 `sq_sum`。
2. **优化访存形式：** 对 `x`、`weight` 使用 `__ldg`；在 `cols % 4 == 0` 且地址对齐时使用 `float4`。

### V2：向量化 + Warp 归约

**文件：** `rmsnorm_v2.cu`

相对 V1 进一步强调：

- `float4` 向量化加载/写回，降低访存指令数
- `float4` 参与 `sq_sum` 计算与输出写回
- 共享内存 `warp_sums[]` 等完成跨 warp 的平方和归约

### V3：融合 Kernel

**文件：** `rmsnorm_v3.cu`

在向量化与并行写回基础上融合路径：

- 归约与输出阶段尽量全线程参与
- 可选用 `__fmul_rn` 等内建保证舍入行为
- 面向较大 hidden dim 的带宽友好访问模式

### cuDNN / CUB 参考（聚合基准内）

**文件：** `rmsnorm_benchmark_all.cu`

基准程序内联调用 **CUB**（归约等）与 **cuDNN**（RMSNorm 路径）作为库对照，与自研 kernel 同表计时。

## 横向对比

**文件：** `rmsnorm_benchmark_all.cu`

聚合多版本与 CUB、cuDNN 的计时对比，结果写入 `data/results/rmsnorm_all_comparison.csv`（需本地运行生成）。

## 性能结果

### 执行时间对比 (ms)

| 矩阵规模 | V0 Naive | V1 WarpReduce | V2 Vectorized | CUB | CPU |
|----------|----------|---------------|---------------|-----|-----|
| 128×128 | 0.032 | 0.021 | 0.018 | 0.086 | 0.016 |
| 256×256 | 0.087 | 0.019 | 0.018 | 0.034 | 0.062 |
| 512×512 | 0.169 | 0.031 | 0.023 | 0.017 | 0.229 |
| 1024×1024 | 0.332 | 0.089 | 0.054 | 0.052 | 0.640 |
| 4096×4096 | 1.363 | 1.290 | 0.619 | 0.470 | 12.550 |

### 吞吐量对比 (GFLOPS)

| 矩阵规模 | V0 Naive | V1 WarpReduce | V2 Vectorized | CUB |
|----------|----------|---------------|---------------|-----|
| 128×128 | 128.4 | 195.7 | 228.4 | 47.9 |
| 256×256 | 235.6 | 1,086.4 | 1,163.7 | 605.3 |
| 512×512 | 607.8 | 3,278.9 | 4,468.4 | 6,035.6 |
| 1024×1024 | 648.4 | 2,419.8 | 3,982.6 | 4,153.1 |
| 4096×4096 | 1,009.1 | 1,065.9 | 2,220.4 | 2,923.4 |

### 关键结论

1. **V2 Vectorized** 在大矩阵场景表现最佳，4096×4096 达到 **2,220 GFLOPS**
2. **V1 WarpReduce** 在中小矩阵表现优秀，256×256 达到 **1,086 GFLOPS**
3. **CUB** 在大矩阵场景有优势，但小矩阵开销较大
4. **CPU** 在小矩阵极快，不适合大矩阵批量处理

## 产物路径

- 可执行文件：`build/bin/`（目标名以 `rmsnorm/CMakeLists.txt` 为准，例如 `rmsnorm_v0_naive`、`rmsnorm_v1`、`rmsnorm_benchmark_all` 等）
- 源文件：`rmsnorm_v0_naive.cu` … `rmsnorm_v3.cu`、`benchmark_all.cu`
- 性能 CSV：`data/results/`

## NVIDIA 参考 API

- **cuDNN：** `cudnnNormalizationForward`（`CUDNN_NORM_RMS`）
- **cuBLAS / 自定义：** 亦可拆成平方和归约 + 逐元素缩放实现
