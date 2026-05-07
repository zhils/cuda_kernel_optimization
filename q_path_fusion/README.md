# Q 路径融合（Q Path Fusion）

## 1. 目标

将注意力模块中 Query 分支的两步计算融合到一条路径：

1. `RMSNorm`
2. `Linear(Q)`（即乘 `W_q` 加 `b_q`）

融合后减少中间张量回写，便于后续继续做向量化、warp/block 归约、分块 GEMM 等优化。

## 2. 数学表达式

设输入为 `X \in R^{R x D}`，其中 `R` 是行数（token 数），`D` 是隐藏维度。  
`gamma \in R^D` 为 RMSNorm 权重，`W_q \in R^{D x D}`，`b_q \in R^D`。

对第 `r` 行：

```text
s_r = sqrt( (1 / D) * sum_{k=0}^{D-1} X_{r,k}^2 + eps )
N_{r,k} = (X_{r,k} / s_r) * gamma_k
Q_{r,n} = sum_{k=0}^{D-1} N_{r,k} * W_{q,k,n} + b_{q,n}
```

矩阵形式可写为：

```text
N = RMSNorm(X; gamma, eps)
Q = N * W_q + b_q
```

## 3. 版本规划

- `v0`: 朴素融合版（正确性基线）
- `v1`: 访存优化版（并行归约 + 共享内存缓存 `norm`）
- `v2`: 在 `v1` 基础上改为 warp-per-row（`block=128`，每 lane 同时算 2 个输出）
- `v2_bq_only`: `v2` 对照版（只缓存 `bq`，`wq` 直接访存；每 lane 同算 2 输出）
- `v3`: 在 `v2` 基础上增加 `wq` tile 双缓冲（ping-pong）

