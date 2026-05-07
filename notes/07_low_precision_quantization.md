# 低精度量化：损失补偿与精度-性能权衡策略

> 本文档回答：低精度（INT8/FP8/FP4）的量化损失怎么补偿？怎么兼顾精度和性能？

---

## 1. 量化的本质问题

### 1.1 数学定义

```
量化：X_fp32 → X_int8 = round(X_fp32 / scale) + zero_point
反量化：X_fp32 ≈ (X_int8 - zero_point) × scale

scale = (max_val - min_val) / (2^bits - 1)
```

### 1.2 精度损失的三个来源

| 来源 | 严重程度 | 典型误差量级 |
|------|----------|-------------|
| **截断误差**（clipping） | 🔴 严重 | 离群值直接被截断，信息完全丢失 |
| **舍入误差**（quantization error） | 🟡 中等 | ~scale/2 per element，通常在 1e-3~1e-1 |
| **累加误差**（accumulation drift） | 🟡 中等 | 随累加次数 N 增长，~O(√N) |

### 1.3 为什么推理场景中精度损失通常可接受？

```
在大模型推理中，量化精度损失往往 < 1% perplexity 退化，原因：
  1. 模型训练时的过参数化 → 有冗余度
  2. 推理是前向过程，没有梯度累积误差
  3. 大量的 layer norm 和 residual 连接帮助稳定数值分布
```

---

## 2. 补偿策略图谱

```
精度要求高                             追求极限性能
   │                                      │
   ├─ INT8 weight-only 量化               ├─ INT4 weight + INT8 activation
   ├─ Per-channel scaling                 ├─ Group-wise quantization (group=128)
   ├─ SmoothQuant（activation 平滑）      ├─ GPTQ / AWQ（权重重要性排序）
   ├─ FP8 (E4M3/E5M2)                    ├─ FP4（Blackwell 原生支持）
   └─ BF16 训练 + INT8 推理               └─ KV Cache INT8/FP8 量化
```

---

## 3. 补偿策略详解

### 3.1 Strategy 1: Per-Channel Scaling（最基础，必做）

```cpp
// ❌ Per-Tensor 量化：所有通道共用一个 scale
// 问题：通道间数值范围差异大（可达 10×）
//      → 小范围的通道信息被淹没
float scale_tensor = max_abs(weight) / 127.0f;
int8_t w_quant = round(weight / scale_tensor);

// ✅ Per-Channel 量化：每个输出通道独立 scale
// 精度提升：通常比 per-tensor 好 2-5% 准确率
for (int oc = 0; oc < out_channels; oc++) {
    float scale_channel = max_abs(weight[oc]) / 127.0f;
    for (int ic = 0; ic < in_channels; ic++) {
        w_quant[oc][ic] = round(weight[oc][ic] / scale_channel);
    }
}
```

**在 GEMM kernel 中使用 per-channel scale：**

```cpp
// INT8 GEMM with per-channel scaling
int acc = 0;
for (int k = 0; k < K; k += 4) {
    acc = __dp4a(packed_a, packed_b, acc);
}
// 反量化：每个输出列用独立的 B_scale[col]
// 注意：B 的 scale 是 per-output-channel（per-column）
float result = static_cast<float>(acc) * A_row_scale * B_col_scale[col];
//                                           ^^^^^^^^^^^^^^^^
//                                           per-channel
```

### 3.2 Strategy 2: SmoothQuant（解决 activation 离群值问题）

```
问题：
  Activation 中有"离群通道"——某些通道的值比其他通道大 5-10×
  → per-tensor 量化时，scale 被离群通道主导 → 正常通道信息被压缩

SmoothQuant 思路：
  把 activation 的量化难度"平滑地"转移到 weight 上

  原始：                SmoothQuant：
  Y = X × W            Y = (X × diag(s)) × (diag(s)^{-1} × W)
                           ^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^
                           X'更容易量化      W'仍然易量化

  s_j = max(|X_j|)^α / max(|W_j|)^{(1-α)}
  
  其中 α ∈ [0.5, 0.7] 是迁移因子
  α=0.5: 两边的难度均匀分配
  α=0.7: 更激进地迁移（适合 activation 离群值特别大的模型）
```

```python
# SmoothQuant 实现（简化版）
def smooth_quant_scale(X_sample, W, alpha=0.5):
    # X_sample: [batch, seq_len, hidden] 的 calibration 数据
    # W: [hidden, out_features]
    
    max_x = torch.max(torch.abs(X_sample), dim=(0, 1))[0]    # per-channel max
    max_w = torch.max(torch.abs(W), dim=1)[0]                 # per-input-channel max
    
    s = torch.pow(max_x, alpha) / torch.pow(max_w, 1 - alpha)
    
    X_smoothed = X_sample * s
    W_smoothed = W / s[:, None]
    
    return X_smoothed, W_smoothed
```

### 3.3 Strategy 3: GPTQ（权重按重要性量化）

```
核心思想：不是一次性量化所有权重，而是逐列量化，每量化一列后
        用 Hessian 信息补偿其余列的误差

算法流程：
  1. 计算权重的 Hessian 矩阵：H = 2X^TX（来自 calibration 数据）
  2. For each column j:
     a. 量化 W[:, j] → W_quant[:, j]
     b. 计算误差：δ = (W[:, j] - W_quant[:, j]) / H[j, j]
     c. 补偿剩余列：W[:, j+1:] -= δ × H[j, j+1:]
  3. 最终所有列都被量化，误差尽可能被后续列吸收

复杂度：O(d_out × d_in^2)，对超大模型可能较慢
优点：4bit 量化后精度损失通常 < 1% perplexity
```

### 3.4 Strategy 4: AWQ（Activation-Aware Weight Quantization）

```
核心发现：不是所有权重对精度同等重要
         → 某些权重对应"重要通道"（activation 范数大的通道）

AWQ 做法：
  1. 统计每个 input channel 的 activation 平均范数
  2. 对重要通道的权重，量化前先"放大"（scaling up），减小相对量化误差
  3. 对应的 activation 通道缩小同样的倍数（在下一个 layer norm 中吸收）

优势：
  - 比 GPTQ 简单（不需要 Hessian 计算）
  - 保持 per-channel 对称量化
  - 4bit 量化精度与 GPTQ 相当
```

### 3.5 Strategy 5: 混合精度量化

```
不是所有层都能用同样的低精度

混合精度策略：
  Layer 0-1 (embedding):           FP16（输入分布变化大）
  Attention QKV projection:        INT8 per-channel
  Attention output projection:     INT4 + GPTQ/AWQ
  FFN gate & up:                   INT8 per-channel
  FFN down:                        INT4 + GPTQ/AWQ
  Layer Norm / RMSNorm:            FP16（对分布变化敏感）
  Final LM head:                   FP16（直接决定输出 token 概率）

自动决策方法：
  1. 每层试量化 → 测 perplexity 退化
  2. 退化 > 阈值 → 该层回退到高精度
  3. 最终得到最优的每层精度分配
```

### 3.6 Strategy 6: FP8（E4M3 / E5M2）—— 新趋势

```
FP8 格式（Hopper+ / Ada+ / Blackwell 原生支持）：

  E4M3 (推理): 1 sign + 4 exp + 3 mantissa = 8 bits
    → 范围: ±240（但保留了较高精度）
    
  E5M2 (训练): 1 sign + 5 exp + 2 mantissa = 8 bits
    → 范围: ±57344（动态范围大，适合梯度）

相比 INT8 的优势：
  ✅ 不需要 calibration 数据（浮点格式天然"适应"数据分布）
  ✅ 不需要 scale/zero_point — 直接用 truncation
  ✅ 反量化就是 cast：fp8 → fp16 = 一次 type conversion
  ✅ 动态范围自动调整（指数位自适应）

在 C++ 中：
  cublasLtMatmul with CUBLASLT_MATMUL_DESC_COMPUTE_TYPE=CUBLAS_COMPUTE_32F_FAST_16F
  输入: __nv_fp8_e4m3 类型
```

---

## 4. 精度补偿策略对比

| 策略 | 精度提升 | 性能开销 | 实现难度 | 适用场景 |
|------|----------|----------|----------|----------|
| Per-Tensor → Per-Channel | +2-5% acc | 0%（只是 scale 数组变了） | 🔴 低 | **必做**，任何量化都应使用 |
| SmoothQuant | +3-8% acc | ~1%（额外 scale 乘法） | 🟡 中 | Activation 离群值严重的模型 |
| GPTQ | +5-10% acc@4bit | 0%（仅权重重排） | 🔴 高 | 4bit 权重量化 |
| AWQ | +4-8% acc@4bit | ~1-2%（channel scaling） | 🟡 中 | INT4 推理 |
| 混合精度 | +2-10% acc | 取决于回退比例 | 🟡 中 | 所有场景都适用 |
| FP8 (vs INT8) | +3-5% acc | ~5%（硬件 cast 开销） | 🟢 低 | H100/L20/RTX50系列 |

---

## 5. 精度-性能权衡决策框架

```
问题 1：你的精度目标是什么？
  ├─ < 0.5% perplexity 退化 → 用 FP8 或 INT8 per-channel + SmoothQuant
  ├─ < 1% perplexity 退化   → INT8 per-channel 通常够用
  ├─ < 3% perplexity 退化   → 4bit 量化 + GPTQ/AWQ
  └─ 不限                    → INT4 + 混合精度

问题 2：你的硬件支持什么？
  ├─ H100/H800/B200/L20 → FP8（Transformer Engine 原生加速）
  ├─ RTX 40/50 系列     → FP8（sm_89/sm_120）+ INT8 dp4a
  ├─ A100               → INT8 dp4a / BF16 Tensor Core
  └─ V100/T4            → INT8 dp4a (无 Tensor Core 加速)

问题 3：你做什么任务？
  ├─ LLM 推理     → KV Cache INT8/FP8 + Weight INT4/INT8
  ├─ 图像分类     → INT8 足够（精度敏感度低）
  ├─ 目标检测     → 混合精度（某些层容易退化）
  └─ 科学计算     → 不建议低精度（精度敏感度高）
```

---

## 6. 在 CUDA kernel 中实现精度补偿

```cpp
// ====== 带精度补偿的 INT8 GEMM kernel ======

__global__ void Int8GemmWithCompensation(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    const float* __restrict__ A_scale,     // per-row (或 per-block)
    const float* __restrict__ B_scale,     // per-column
    const float* __restrict__ residual,    // 残差补偿项（GPTQ 风格）
    float* __restrict__ C,
    int M, int N, int K) {

    // ... GEMM 循环 ...

    // 反量化 + 补偿
    float result = static_cast<float>(acc) * A_row_scale * B_col_scale;

    // 精度补偿 1：Per-Channel Bias Correction
    // 量化前计算 bias = mean(W_fp32 - W_quant_dequant, dim=input_channels)
    // 反量化后加回 bias
    result += bias_correction[col];

    // 精度补偿 2：Residual（GPTQ 预留的补偿张量）
    if (residual != nullptr) {
        // residual 是 GPTQ 在量化过程中累积的未吸收误差
        // 在 kernel 中直接加到结果上
        result += residual[row * N + col];
    }

    // 精度补偿 3：Clipping（防止离群值破坏后续层）
    // 可选：把反量化结果 clamp 到量化前统计的范围
    // result = fminf(fmaxf(result, output_min), output_max);

    C[row * N + col] = result;
}
```

---

## 7. 面试应答模板

**面试官："低精度量化有精度损失，你怎么补偿？"**

> "量化精度损失有三个来源：截断、舍入、累加。我的补偿策略是分层级的：
>
> 第一层——基础补偿，所有量化都应该做。per-channel scaling 替代
> per-tensor，精度直接提升 2-5%，几乎没有性能开销。
>
> 第二层——激活平滑。如果发现某些 activation 通道有离群值（常见的
> 是大模型中间层的某些通道值比其他大 5-10×），我会用 SmoothQuant，
> 把量化的难度从 activation 向 weight 平滑迁移。数学上就是把
> activation 乘 diag(s)、weight 除 diag(s)，s 的选择基于 activation
> 和 weight 的统计量。
>
> 第三层——权重重要性保护。对于 4bit 量化，GPTQ 和 AWQ 是两个主流方案。
> GPTQ 用 Hessian 信息做逐列量化+误差补偿，精度损失通常 < 1% perplexity。
> AWQ 更简单——找到重要的通道，量化前放大、下一个 norm 层吸收，效果
> 和 GPTQ 相当但实现简单得多。
>
> 实际选择取决于精度目标和硬件支持。如果有 FP8 支持的硬件（H100/L20/
> RTX 50 系列），我会优先用 FP8，因为它不需要 calibration 也不需要
> 额外的补偿——浮点格式的指数位天然适应数值分布变化。"

---

## 8. 速查表：量化类型与补偿方案

| 量化格式 | 典型精度损失 | 推荐补偿方案 | 硬件要求 |
|----------|-------------|-------------|----------|
| FP16 → BF16 | 几乎无损 | 不需要 | sm_80+ |
| FP32 → FP16 | < 0.1% | 不需要（loss scaling 训练时） | sm_60+ |
| FP16 → INT8 (per-channel) | 0.2-0.5% | Per-Channel Scaling | sm_61+ |
| FP16 → INT8 (per-tensor) | 1-3% | SmoothQuant | sm_61+ |
| FP16 → INT4 (naive) | 5-15% | GPTQ / AWQ | sm_61+ |
| FP16 → INT4 (with GPTQ) | 1-2% | GPTQ + 混合精度 | sm_61+ |
| FP16 → FP8 (E4M3) | 0.1-0.3% | 通常不需要 | sm_89+ |
| KV Cache FP16→INT8 | 0-0.5% | Per-channel + 少量校准 | sm_61+ |
| KV Cache FP16→FP8 | < 0.1% | 不需要 | sm_89+ |
