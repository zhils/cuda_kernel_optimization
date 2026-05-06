# GitHub 仓库内容与上传约定

本文说明本仓库**适合公开托管的内容**、**建议不要提交的内容**，以及协作者克隆后需要注意的事项。

---

## 1. 建议提交并长期维护的内容（适合 push 到 GitHub）

| 类别 | 说明 |
|------|------|
| **CUDA 源码** | 各子目录下的 `*.cu`、公共头文件与实现（如 `common/include/`、`common/src/`）。 |
| **构建脚本** | 顶层与各子项目的 `CMakeLists.txt`；若有 `cmake/` 工具链片段一并保留。 |
| **文档** | `README.md`、`docs/*.md`、各算子目录下的 `README.md`（方法、版本演进、复现步骤）。 |
| **小型配置** | 不含有本机隐私信息的通用配置；**不要**提交带个人路径或许可证密钥的私有 overlay。 |

原则：**可复现的学习型代码 + 说明如何在你自己的机器上配置 CUDA/cuDNN**。

---

## 2. 建议不要提交（应留在本机或由 CI 生成）

以下内容体积大、环境相关或易过期，已在根目录 `.gitignore` 中尽量覆盖；若本地仍有残留，上传前可删掉或不要 `git add`。

| 类别 | 典型路径 / 模式 | 原因 |
|------|-----------------|------|
| **构建目录与二进制** | `build/`、`*.exe`、`*.obj`、`*.lib`、`*.exp` | 体积大、平台相关；他人用 CMake 自行生成即可。 |
| **CMake 生成物** | `CMakeCache.txt`、`CMakeFiles/`、`*.cmake`（生成在 build 内即可） | 不应进源码树。 |
| **剖析与 trace** | `*.ncu-rep`、`*.nsys-rep`、大型 `*.sqlite` | Nsight 输出可能很大且含环境信息。 |
| **基准原始数据（可选）** | `data/*.csv`、`data/results/`、`*.npy` | 若体积大或含内部用例，可不跟踪；需要「可复现数字」时再在文档中写命令生成。 |
| **反汇编 / ISA 导出** | `gemm/asm/` 下 `ptx/`、`sass/` 等（由自定义 target 生成） | 可完全由 nvcc 从 `.cu` 再生，避免仓库膨胀。 |
| **IDE 私有文件** | `.vs/`、`.vscode/`（除非团队约定提交共享 `settings.json` 模板）、`*.user` | 个人本机路径。 |
| **系统杂项** | `.DS_Store`、`Thumbs.db` | 无意义 diff。 |

**截图（`*.png`）**：若仅作博客配图且体积很小，可以提交；若批量导出、体积大，建议本地保留或放 Release 附件，不必全部进 Git。

---

## 3. 敏感与合规（必须注意）

- **不要**提交 API Key、云凭证、内网地址、**带许可证的 cuDNN 二进制本身**（遵守 NVIDIA 软件许可条款）。
- 当前部分 `CMakeLists.txt` 中可能含有 **本机绝对路径**（例如 cuDNN 安装目录）。公开仓库时建议改为：
  - 通过 `-DCUDNN_ROOT=...` 或环境变量传入，或  
  - 在 `README` / `docs/benchmark_environment.md` 中说明由用户自行修改路径。  
  上传前用 `git grep "C:/Program"` 等自检一遍。

---

## 4. 克隆仓库后的最小复现步骤（给协作者，Windows）

1. 安装与项目目标架构匹配的 **CUDA Toolkit**（见根目录 `README.md` 与 `docs/benchmark_environment.md`）。
2. 若构建 **softmax / rmsnorm** 的 cuDNN 参考或 `*_benchmark_all`，需安装 **cuDNN**，并按 CMake 要求配置 include/lib 路径（或后续改为 CMake 变量）。
3. 在仓库根目录打开 **PowerShell**：创建 `build` 目录，`cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=<你的 GPU 对应数值>`，再 `cmake --build . --config Release`；可执行文件一般在 `build\bin\Release\`。

---

## 5. 可选：用 Git LFS

若未来需要跟踪 **大体积** 剖析报告或数据集，再考虑 [Git LFS](https://git-lfs.github.com/)，默认不必开启。

---

## 6. 与「论文级复现」相关的建议

若希望仓库在 GitHub 上「数字也可复现」：

1. 在文档中写明：**CUDA 驱动版本、GPU 型号、`CMAKE_CUDA_ARCHITECTURES`**。  
2. 性能表注明**测量日期**；大表可放到 `docs/` 或 Wiki，避免主 `README` 过长。  
3. 对 `data/`：要么提供**小型公开用例** CSV，要么在代码里内置默认小规模 case，避免依赖缺失导致首次运行失败。

以上约定与根目录 `.gitignore` 一致；若团队策略不同，可只改 `.gitignore` 并在本文同步说明。

---

## 7. CMake 目标与源文件一致

`gemm/CMakeLists.txt` 中的 `GEMM_VARIANTS` 列表必须与 `gemm/` 目录下存在的 `同名.cu` 一一对应；若列表里含有实验目标而仓库未包含对应 `.cu`，他人克隆后将无法配置通过。公开前请删除未跟踪的变体或补全源文件。
