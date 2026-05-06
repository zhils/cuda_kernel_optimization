# CUDA Kernel 优化

本项目逐步优化深度学习核心算子：从朴素基线到更高利用率的 CUDA Kernel，并在 **NVIDIA GeForce RTX 5060 Ti（Blackwell，sm_120）** 上与 cuBLAS / cuDNN 做对比实验。

**环境与文档约定：** 命令与路径默认 **Windows 11**、**PowerShell**、**Visual Studio 多配置生成器**；可执行文件一般在 `build/bin/Release/`。若使用 **Ninja** 等单配置生成器，可执行文件通常在 `build/bin/`（无 `Release` 子目录）。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [gemm/README.md](gemm/README.md) | GEMM 版本演进、可执行文件对照、`gemm_benchmark_all` 范围说明 |
| [softmax/README.md](softmax/README.md) | Softmax 各版本与参考实现 |
| [rmsnorm/README.md](rmsnorm/README.md) | RMSNorm 各版本与参考实现 |
| [docs/benchmark_environment.md](docs/benchmark_environment.md) | 硬件、软件版本与复现命令（Windows） |
| [docs/github_repository_guidelines.md](docs/github_repository_guidelines.md) | 适合提交到 GitHub 的内容与 `.gitignore` 约定 |

---

## 性能总览

<!-- 在目标 GPU 上跑完 benchmark 后可更新具体数字 -->

| 算子 | 瓶颈类型 | 说明 |
|------|----------|------|
| **GEMM** | 计算为主 | 演进见 [gemm/README.md](gemm/README.md)；`gemm_benchmark_all` 对比 V0–V3 与 **FP32 `cublasSgemm`** |
| **Softmax** | 内存为主 | 见 [softmax/README.md](softmax/README.md) |
| **RMSNorm** | 内存为主 | 见 [rmsnorm/README.md](rmsnorm/README.md) |

---

## 优化方法论

每个算子遵循同一套流程：

```
理论分析 → 基线实现 → 性能剖析（Nsight）→ 定向优化 → 正确性验证 → 迭代
```

1. **Roofline**：算术强度，判断计算/访存主导  
2. **朴素基线**：正确、未调优  
3. **Nsight Compute**：stall、带宽、占用率、指令构成  
4. **渐进版本**：每版针对一个瓶颈  
5. **验证**：相对 CPU 或库参考的误差与计时  

---

## 算子与版本速览

### GEMM（通用矩阵乘）

[详细说明 →](gemm/README.md)

| 版本 | 优化点 | 关键技术 |
|------|--------|----------|
| V0 | 朴素基线 | 每线程一个输出元素 |
| V1 | 共享内存分块 | 16×16 tile，`float4` / `__ldg` |
| V2 | 线程级寄存器分块 | 每线程 **8×8** 子块，CTA **128×128**，`kTileK=16` |
| V3 | 双缓冲 | 在 V2 几何上双缓冲 A/B SMEM，重叠加载与计算 |
| V4 | 大 tile FP32（主线） | 128×128 CTA，`kTileK=32`，`__launch_bounds__(256,2)` |
| Ref | cuBLAS Tensor | `gemm_cublas_ref`：`cublasGemmEx`，FP16 输入、FP32 累加 |

**注意：** `gemm_benchmark_all.exe` **仅**汇总 **V0–V3** 与 **`cublasSgemm`**；不包含独立 **`gemm_v4`** 与 **`gemm_cublas_ref`**，需分别运行对应 `.exe`。

### Softmax

[详细说明 →](softmax/README.md)

| 版本 | 优化点 | 关键技术 |
|------|--------|----------|
| V0 | 朴素 | 每行单线程，3-pass |
| V1 | Block 归约 | 共享内存 max / sum |
| V2 | Online softmax | 单遍 Milakov 风格 |
| V3 | Warp shuffle + 向量化 | `__shfl_sync`，`float4` |
| Ref | cuDNN | `cudnnSoftmaxForward` |

### RMSNorm

[详细说明 →](rmsnorm/README.md)

| 版本 | 优化点 | 关键技术 |
|------|--------|----------|
| V0 | 朴素 | 每行单线程 |
| V1 | Warp 归约 | `__shfl_sync` 求 sq_sum |
| V2 | 向量化 | `float4` + 跨 warp 归约 |
| V3 | 融合写回 | 全线程参与输出 |
| Ref | cuDNN | `cudnnNormalizationForward` |

---

## 构建与运行

在**仓库根目录**打开 **PowerShell**：

```powershell
# 构建（示例：sm_120；请按本机 GPU 修改 CMAKE_CUDA_ARCHITECTURES）
New-Item -ItemType Directory -Force build | Out-Null
Set-Location build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build . --config Release
Set-Location ..

# 单个可执行文件（Nsight 或调试）
.\build\bin\Release\gemm_v0.exe
.\build\bin\Release\gemm_v3.exe

# GEMM 汇总（V0–V3 + cuBLAS FP32）；V4 / Tensor 参考需单独运行
.\build\bin\Release\gemm_benchmark_all.exe
.\build\bin\Release\gemm_v4.exe
.\build\bin\Release\gemm_cublas_ref.exe

# 其他算子汇总（需 cuDNN 路径在 CMake 中配置正确）
.\build\bin\Release\softmax_benchmark_all.exe
.\build\bin\Release\rmsnorm_benchmark_all.exe

# Nsight Compute 示例（需安装 NCU）
ncu --set full .\build\bin\Release\gemm_v2.exe
```

**softmax / rmsnorm** 若 CMake 中写死本机 cuDNN 路径，克隆后需在本地修改或改为 CMake 变量；见 [docs/github_repository_guidelines.md](docs/github_repository_guidelines.md)。

---

## 开源与 GitHub

- 提交范围、忽略规则与合规注意： **[docs/github_repository_guidelines.md](docs/github_repository_guidelines.md)**  
- 概要：提交源码、`CMakeLists.txt`、文档；不要提交 `build/`、大体积剖析报告、生成 ISA 目录（如 `gemm/asm/`）及含密钥或个人路径的配置。

---

## 环境信息（参考）

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB |
| 架构 | Blackwell（sm_120） |
| CUDA Toolkit | 以本机安装为准（文档示例曾用 13.x） |
| cuBLAS / cuDNN | 随 Toolkit / 独立安装包 |
| 计时 | 各子项目 `common::` 与 `cudaEvent` 组合（见各 `*.cu`） |

更细的版本表见 [docs/benchmark_environment.md](docs/benchmark_environment.md)。

---

## 项目结构

```
├── common/              通用基准与工具（头文件 + 源文件）
├── gemm/                GEMM：v0–v4、cuBLAS 参考、benchmark_all
├── softmax/             Softmax：v0–v3、cuDNN 参考、benchmark_all
├── rmsnorm/             RMSNorm：v0–v3、cuDNN 参考、benchmark_all
├── docs/                环境、复现、GitHub 约定等
├── data/                测试用例 / 结果 CSV（默认可能被 gitignore）
└── CMakeLists.txt       顶层构建入口
```
