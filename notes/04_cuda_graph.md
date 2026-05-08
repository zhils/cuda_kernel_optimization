# CUDA Graph：减少 Kernel Launch Overhead

## 问题

每次 `<<<grid, block>>>` 都需要 CPU 向 GPU 提交一次 kernel launch。
对于小 kernel，launch overhead（~3-10μs）可能超过 kernel 本身执行时间。

在 `fused_conv1d_silu_v0` 中，5 次 kernel launch = 5× launch overhead。
在 Transformer 推理中，每个 layer 可能有数十个小 kernel → launch overhead 累积显著。

## CUDA Graph 解决什么

CUDA Graph 将**一系列 kernel launch 捕获为一个图（graph）**，一次提交，GPU 按依赖关系自动调度：

```
传统方式:               CUDA Graph:
  Launch A  ← overhead     Capture graph { A → B → C }  ← 一次性捕获
  Launch B  ← overhead     Launch graph                 ← 一次 overhead
  Launch C  ← overhead
```

对于包含大量小 kernel 的场景（如 B=8 L=2048 的 fused_conv1d_silu），CUDA Graph 能显著降低 launch overhead 占用的比例。

## 核心 API

```cuda
// Step 1: 创建 graph
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// — 以下操作被捕获到 graph 中 —
kernel_a<<<grid, block, 0, stream>>>(...);
kernel_b<<<grid, block, 0, stream>>>(...);
kernel_c<<<grid, block, 0, stream>>>(...);
// ———————————————————

cudaStreamEndCapture(stream, &graph);

// Step 2: 实例化（编译 graph 为可执行形式）
cudaGraphExec_t exec;
cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);

// Step 3: 重复执行（零 overhead）
for (int i = 0; i < 1000; ++i) {
    cudaGraphLaunch(exec, stream);  // 一次提交，所有 3 个 kernel 全部 launch
}
```

## 什么场景收益最大

| 场景 | 有多少 kernel | launch overhead 占比 | Graph 收益 |
|------|:------------:|:--------------------:|:----------:|
| GEMM 4096³ | 1 个大 kernel | <1% | 几乎无收益 |
| Fused Conv1D+SiLU v0 | 5 个小 kernel | 5-15% | 中等收益 |
| **Transformer Layer** | **10-30 个小 kernel** | **30-50%** | **高收益** |
| BERT Inference | 成百上千个 kernel | 显著 | **非常适合** |

**注意**：fused_conv1d_silu_v2 已经将 5 个 kernel 融合为 2 个，CUDA Graph 对它的边际收益较小。但如果你有分离版本的完整 Transformer Layer，CUDA Graph 就非常有价值。

## 在 fused_conv1d_silu_v0 上的应用示例

```cuda
void RunWithGraph(dim3 grid, dim3 block, ...) {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    
    // 单次捕获
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    RunGpuV0(d_x, d_W_qkv, ..., B, L, D, H, k_size);  // 捕获 5 个 launch
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
    
    // 零 overhead 重复执行
    for (int i = 0; i < kRepeat; ++i) {
        cudaGraphLaunch(exec, stream);
    }
    cudaDeviceSynchronize();
    
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
}
```

## 注意事项

1. **图一次捕获，多次执行** —— 捕获期间的内存分配/数据拷贝也被固定下来
2. **动态分支** —— graph 中的 kernel 不能有依赖于运行时的分支
3. **cudaMemset/cudaMemcpy 也能被捕获** —— 整个 pipeline 都可以 graph 化
4. **NVIDIA 推荐组合方式**：CUDA Graph + MPS（Multi-Process Service）用于生产推理

## 参考

- [CUDA Graph 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- GTC 演讲: "CUDA Graphs: A New Way to Submit Work to the GPU"
