# 算子融合判断标准与评估方法

## 1. 什么是算子融合

算子融合是将多个连续的 CUDA kernel 合并为一个 kernel，中间结果不写回全局内存，而是留在寄存器/共享内存中。

```
融合前:
Kernel1: 读A → 计算 → 写B (全局内存)
Kernel2: 读B → 计算 → 写C (全局内存)
Kernel3: 读C → 计算 → 写D (全局内存)

融合后:
FusedKernel: 读A → 计算 → B(寄存器) → 计算 → C(寄存器) → 计算 → 写D(全局内存)
```

## 2. 什么样的算子可以融合？

### 2.1 可以融合的特征（必要条件 + 充分条件）

| 条件 | 说明 | 示例 |
|------|------|------|
| **数据依赖链**（必要） | 前一个算子的输出是后一个的输入 | Linear → SiLU → RMSNorm |
| **Element-Wise 操作** | 每个线程独立计算自己的元素，无需通信 | ReLU, sigmoid, add, mul, mul+add |
| **相同的数据布局** | 输入输出维度相同，无 reshape/transpose | (B,L,H)→(B,L,H) 的连续操作 |
| **Fusion 不破坏并行度** | 融合后每个线程仍有足够工作量 | 不能把 1M 线程的 kernel 融合成 10 个线程 |
| **无跨元素依赖** | 不需要等其他元素结果（除非通过共享内存归约） | RMSNorm 的归约可以融合，但需要 sync |

### 2.2 禁止融合的特征（红线）

| 条件 | 原因 |
|------|------|
| **中间结果需要被多个后续算子使用** | 融合后 B 消失，其他算子拿不到 |
| **不同大小的数据** | reshape / slice / gather 导致线程映射复杂 |
| **中间结果的 Shape 会变** | 融合后不好分配寄存器/smem |
| **跨 Batch 依赖** | batch norm 需要跨 batch 统计 → 必须分离 |
| **SMEM 超限** | 融合后 smem 不足 → occupancy 降到 0 |
| **寄存器超限** | 编译后 spills to local memory → 性能暴跌 |
| **包含异步操作** | cudaStream 依赖 → 不能合并到一个 kernel |
| **有全局同步** | cooperative_groups::grid_group sync → 不能合并 |

### 2.3 具体案例分析

#### ✅ 可以融合：Linear + SiLU + RMSNorm + Mul + Linear

```cpp
// 原始 5 个 kernel
step1: x @ W1 + b1 → h1              (GEMV, B×L×H elements)
step2: SiLU(h1) → h2                 (element-wise)
step3: RMSNorm(h2, γ) → h3           (element-wise with reduction)
step4: h3 ⊙ h2 → h4                  (element-wise)
step5: h4 @ W2 + b2 → output         (GEMV)

// 融合后：1 个 kernel
// 每个 (b,t) 的线程块：
//   1. 加载 x_bt 到寄存器
//   2. 计算 h1 = GEMV(x_bt, W1) + b1  [寄存器]
//   3. 计算 h2 = SiLU(h1)              [寄存器]
//   4. 归约求 rms(h2)                   [shmem + warp shuffle]
//   5. 计算 h3 = h2 / rms * γ          [寄存器]
//   6. 计算 h4 = h3 * h2               [寄存器]
//   7. 计算 output = GEMV(h4, W2) + b2 [寄存器 → 写回 global]
// 中间变量全部在寄存器中，只读写各一次全局内存
```

#### ✅ 可以融合：4 个 Linear（共享输入）

```cpp
// QKV projection + ZAB projections → 同一个 x 被复用 6 次
// 融合后：
__shared__ float s_x[1024];  // 一次加载 x，所有线程复用
for (int d = tx; d < D; d += 256) s_x[d] = x_bt[d];
__syncthreads();

// 计算 3H + 3×H = 6H features，每个 thread 负责多个 h
float acc_qkv_0 = b_qkv[h0], acc_qkv_1 = b_qkv[h1];
float acc_z = b_z[h], acc_a = b_a[h], acc_b = b_b[h];
for (int d = 0; d < D; d++) {
    float xd = s_x[d];
    acc_qkv_0 += xd * W_qkv[d * 3H + h0];
    acc_qkv_1 += xd * W_qkv[d * 3H + h1];
    acc_z += xd * W_z[d * H + h];
    acc_a += xd * W_a[d * H + h];
    acc_b += xd * W_b[d * H + h];
}
qkv[h0] = acc_qkv_0; qkv[h1] = acc_qkv_1;
z[h] = acc_z; a[h] = acc_a; b[h] = acc_b;
```

#### ❌ 不能融合：Softmax(Q·K^T) + ⊙V

```cpp
// Attention 的核心：Q·K^T 的结果是一个 (N,N) 矩阵
// Softmax 需要遍历整行找 max
// ⊙V 需要 Softmax 的完整结果
// 这三个操作数据依赖紧密，但 Q·K^T 的结果 (N,N) 太大
// 无法完全放在 smem 中 → 必须写回 global memory
// → 不能完全融合，只能做 FlashAttention 式的不完全融合
```

#### ❌ 不能融合：LayerNorm + Linear（如果中间结果被其他分支使用）

```cpp
// 如果 x_norm = LayerNorm(x) 同时用于两个分支：
// 分支1：x_norm @ W1 + b1 → output1
// 分支2：x_norm @ W2 + b2 → output2
// 则 LayerNorm 不能融合到任何一个分支中
// 因为融合后 x_norm 在寄存器中消失了，另一个分支拿不到
```

## 3. 如何评估融合前后的性能收益？

### 3.1 理论计算方法

**Step 1：计算融合前的内存流量**

```
Memory_Read_Before  = Σ(每个 kernel 的输入数据量)
Memory_Write_Before = Σ(每个 kernel 的输出数据量)
Total_Mem_Before    = Read + Write
```

**Step 2：计算融合后的内存流量**

```
Memory_Read_After   = 第一个 kernel 的输入 + 融合后仍需从 global 读取的
Memory_Write_After  = 最后一个 kernel 的输出
Total_Mem_After     = Read + Write
```

**Step 3：计算 Bandwidth Savings**

```
Savings = Total_Mem_Before - Total_Mem_After
Ratio   = Total_Mem_After / Total_Mem_Before
```

**Demo：fused_output_norm_gate 的 4 步融合评估**

```
// 参数：B=8, L=2048, H=256, D_out=512
// Step 1: Gate Projection
//   读 x: 8×2048×256×4B = 16 MB
//   读 W_gate: 256×256×4B = 256 KB
//   写 gate: 8×2048×256×4B = 16 MB
// Step 2: RMSNorm
//   读 gate: 16 MB
//   读 gamma: 256×4B = 1 KB
//   写 x_hat: 16 MB
// Step 3: Multiply Gate
//   读 x_hat: 16 MB
//   读 gate: 16 MB
//   写 y: 16 MB
// Step 4: Linear Output
//   读 y: 16 MB
//   读 W_out: 256×512×4B = 512 KB
//   写 output: 8×2048×512×4B = 32 MB

// 融合前总计：
//   读 = 16+0.256+16+0.001+16+16+16+0.512 = 80.77 MB
//   写 = 16+16+16+32 = 80 MB
//   总计 = 160.77 MB

// 融合后总计（4 步全部融合）：
//   读 x: 16 MB
//   读 W_gate, gamma, W_out（权重忽略不计）
//   写 output: 32 MB
//   总计 ≈ 48 MB

// Savings: 160.77 - 48 = 112.77 MB → 节省 70%
// 但实际上中间步骤的计算量没有减少，
// 所以速度提升 ≈ min(BW_savings, compute_overlap_factor) × launch_overhead_removal
```

### 3.2 实际测量方法

```bash
# Step 1: 测量分离版本的总耗时
# 对每个 kernel 分别计时
ncu --print-summary ./fused_output_norm_gate_v0

# Step 2: 测量融合版本
ncu --print-summary ./fused_output_norm_gate_v1

# Step 3: 对比
# 关注：
#   - Total GPU time
#   - Memory throughput (GB/s)
#   - Kernel launch overhead (多个 kernel vs 1 个 kernel)
```

### 3.3 融合收益量化公式

```
Speedup = T_separate / T_fused

其中：
T_separate = Σ T_kernel_i + (N-1) × LaunchOverhead + N × GlobalMemLatency
T_fused    = T_fused_compute + 1 × LaunchOverhead + 1 × GlobalMemLatency

典型 LaunchOverhead ≈ 5-10 μs per kernel
典型 GlobalMemLatency ≈ 200-800 cycles (HBM)
```

## 4. 如何识别融合机会？

### 4.1 用 Nsight Systems 识别

```bash
# 查看 timeline，找紧密相邻的小 kernel
nsys profile -o timeline ./your_binary
# 在 Nsight Systems GUI 中：
#   - 看 "CUDA HW" 行：如果大量短小的 kernel 挨在一起 → 融合机会
#   - 看 "GPU MemOps"：如果中间 kernel 的输入来自上一个 kernel 的输出 → 融合机会
```

### 4.2 代码层面识别

**识别清单**：

```
□ 代码中有连续多次 kernel launch？
  → 是 → 融合机会
□ 前一个 kernel 的输出紧跟后一个 kernel 的输入？
  → 是 → 融合机会
□ 中间结果不被第三方 kernel 使用？
  → 是 → 融合候选
□ 中间结果的元素之间独立？
  → 是 → 融合复杂度低
□ 总共享内存 < 48KB（或可动态分配）？
  → 是 → 融合可行
□ 总寄存器 < 255 per thread（避免 spills）？
  → 是 → 融合可行
```

---
