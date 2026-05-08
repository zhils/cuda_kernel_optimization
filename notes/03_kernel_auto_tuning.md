# CUDA Kernel Auto-Tuning：自动参数搜索

## 为什么需要 Auto-Tuning？

手写 CUDA kernel 的性能对 tile size、block size、unroll factor 等参数非常敏感。同一 kernel 在不同 GPU 架构上最优参数可能完全不同。

人工手动调参的问题：
- 参数空间大：tile size (多个维度)、block size、unroll depth、prefetch depth
- 经验依赖：能调优但不一定找得到全局最优
- 不可迁移：在 sm_86 上调好的参数，到 sm_120 上可能需要重调

Auto-Tuning 的核心思想：**让计算机替你在参数空间中搜索，实测每个参数组合的性能，返回最优的一组。**

## 一个完整 Auto-Tuner 的设计

```
Tuner 主循环:
  for each 参数组合 (tileM, tileN, tileK, blockSize, ...):
    if 组合合法（共享内存不超限、寄存器不超限）:
      用当前参数编译 kernel
      运行 10 次取中位数
      记录性能
  return 最优参数组合
```

## 可调参数空间（以 GEMM 为例）

| 参数 | 说明 | 搜索范围 | 典型值 |
|------|------|---------|--------|
| `kBlockM` | CTA 在 M 维的 tile 大小 | 32, 64, 128, 256 | 128 |
| `kBlockN` | CTA 在 N 维的 tile 大小 | 32, 64, 128, 256 | 128 |
| `kTileK` | K 维的 tile 大小 | 16, 32, 64 | 16/32 |
| `kThreads` | 线程数 | 128, 256 | 256 |
| `kWarpTilesM` | 每个 warp 的 M-tile 数 | 2, 4 | 4 |
| `kWarpTilesN` | 每个 warp 的 N-tile 数 | 2, 4 | 2 |

## GEMM 参数空间的组合数

```
3 × 3 × 3 × 2 × 2 × 2 = 216 种组合
每次 benchmark 运行 10 次 = 2160 次 kernel launch
每次约 0.1ms = 216ms 搜索
```

对于离线调优，这是完全可以接受的代价。

## 模板化实现（在 C++ 中）

```cuda
// 将 tile size 作为模板参数，而不是 constexpr
template<int BLOCK_M, int BLOCK_N, int TILE_K, int THREADS>
__global__ void GemmKernel(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    // 内核实现与 v4 相同，但使用模板参数
    // ...
}

// 搜索入口
void RunTuner(int M, int N, int K) {
    struct Config {
        int block_m, block_n, tile_k, threads;
        float gflops;
    };
    std::vector<Config> results;
    
    for (int bm : {32, 64, 128, 256}) {
      for (int bn : {32, 64, 128, 256}) {
        for (int tk : {16, 32}) {
          for (int th : {128, 256}) {
            if (bm * bn * tk * sizeof(float) * 2 > 48*1024) continue;  // SMEM bound
            // benchmark and record
          }
        }
      }
    }
}
```

## 实际搜索策略

### 1. 离线 Tuning（编译期）

编译时搜索 + 将结果写入 JSON/CSV：

```
build/tuning_results.json
{
  "4096x4096x4096": {"block_m": 128, "block_n": 128, "tile_k": 16, "threads": 256},
  "1024x1024x1024": {"block_m": 128, "block_n": 128, "tile_k": 16, "threads": 256},
  "512x512x512":   {"block_m": 64,  "block_n": 128, "tile_k": 16, "threads": 128}
}
```

### 2. 在线 Tuning（首次运行自动搜索）

首次运行时自动搜索 → 缓存结果到文件 → 后续直接读取。

## Grid Search vs Random Search

| 方法 | 优点 | 缺点 |
|:---:|------|------|
| **Grid Search** | 保证覆盖所有组合 | 组合数指数增长 |
| **Random Search** | 有限时间内可能找到好解 | 无法保证最优 |
| **Bayesian Optimization** | 高效，收敛快 | 实现复杂 |

对于 CUDA kernel tuning（参数少，通常 ≤6 个），**Grid Search 是最实用的选择**。

## 代码结构建议

```
kernel_auto_tuning/
├── tuner.py           # Python 调优脚本（编译+运行+分析）
├── tuning_results.csv # 调优结果缓存
├── templates/         # 模板化的 CUDA kernel
└── README.md
```

Python 比 CUDA 更适合做调优框架：可以调用 nvcc 编译、解析输出、写 CSV、画图。

## 简历写法

```
• Kernel Auto-Tuning：构建 GEMM 自动参数搜索框架（Python + CUDA），
  在 216 组参数空间中搜索 tile size / block size / thread 的最优组合，
  离线调优结果写入 JSON 缓存，运行时自动加载最优配置。
  相比固定参数配置，在 512³ 规模提升 ~15%。
```
