# PyTorch 自定义算子集成完整指南

> 本文档回答：怎么把 CUDA kernel 集成到 PyTorch 中使用？PyTorch 中你应该了解的全部内容。

---

## 1. PyTorch 自定义算子的三种方式

| 方式 | 难度 | 适用场景 | 例子 |
|------|------|----------|------|
| `torch.utils.cpp_extension.load_inline` | 🔴 最简单 | 快速原型/测试 | 写好 CUDA 源码直接 load |
| `torch.utils.cpp_extension.load` + `setuptools` | 🟡 中等 | 独立库/项目 | 你的 rmsnorm 文件夹完整打包 |
| `torch.library` + `m.def` (TorchScript custom op) | 🟢 正式 | 生产级、需要序列化 | PyTorch 内置算子 |

---

## 2. 快速上手——5 分钟让 RMSNorm 在 PyTorch 中运行

### 2.1 完整的最小可运行示例

```python
# rmsnorm_pytorch_demo.py
import torch
from torch.utils.cpp_extension import load_inline

# ====== 1. CUDA kernel 源码（直接从你的项目中复制） ======
cuda_source = """
#include <cuda_runtime.h>

__global__ void RMSNormV3Kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    int rows, int cols, float eps) {

    extern __shared__ float smem[];
    float* weight_smem = smem;
    float* reduce_smem = smem + cols;

    int r = blockIdx.x;
    int tid = threadIdx.x;

    // 把 weight 加载到共享内存
    for (int i = tid; i < cols; i += blockDim.x) {
        weight_smem[i] = weight[i];
    }
    __syncthreads();

    const float* row_x = x + r * cols;
    float* row_y = y + r * cols;

    float thread_sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = row_x[c];
        thread_sum += v * v;
    }

    reduce_smem[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_smem[tid] += reduce_smem[tid + s];
        __syncthreads();
    }

    float rms = sqrtf(reduce_smem[0] / cols + eps);
    float inv_rms = 1.0f / rms;

    for (int c = tid; c < cols; c += blockDim.x) {
        row_y[c] = row_x[c] * inv_rms * weight_smem[c];
    }
}

// ====== 2. C++ 包装函数（PyTorch 调用入口） ======
torch::Tensor rmsnorm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    float eps) {

    int rows = x.size(0);
    int cols = x.size(1);
    auto y = torch::empty_like(x);

    int threads = std::min(256, static_cast<int>((cols + 31) / 32 * 32));
    if (threads < 32) threads = 32;

    int smem_bytes = cols * sizeof(float) + threads * sizeof(float);
    RMSNormV3Kernel<<<rows, threads, smem_bytes>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        y.data_ptr<float>(),
        rows, cols, eps);

    return y;
}

// ====== 3. 注册为 PyTorch 自定义算子 ======
TORCH_LIBRARY(my_rmsnorm, m) {
    m.def("rmsnorm(Tensor x, Tensor weight, float eps=1e-6f) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_rmsnorm, CUDA, m) {
    m.impl("rmsnorm", rmsnorm_forward);
}
"""

cpp_source = """
#include <torch/extension.h>
torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor weight, float eps);
TORCH_LIBRARY(my_rmsnorm, m) { m.def("rmsnorm", rmsnorm_forward); }
"""

# ====== 4. 编译并加载 ======
module = load_inline(
    name="rmsnorm_extension",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["rmsnorm_forward"],
    extra_cuda_cflags=["-O3"],
    verbose=True
)

# ====== 5. 测试 ======
x = torch.randn(4096, 4096, device="cuda")
weight = torch.ones(4096, device="cuda")

# 调用你的 CUDA kernel！
y = torch.ops.my_rmsnorm.rmsnorm(x, weight, 1e-6)

# 验证正确性
y_ref = x / torch.sqrt((x * x).mean(dim=1, keepdim=True) + 1e-6) * weight
print("Max diff:", (y - y_ref).abs().max().item())
print("Pass!" if (y - y_ref).abs().max() < 1e-3 else "FAIL!")
```

### 2.2 运行

```bash
python rmsnorm_pytorch_demo.py
```

---

## 3. 方式一详解：`load_inline`（快速原型）

### 3.1 完整流程

```python
from torch.utils.cpp_extension import load_inline

module = load_inline(
    name="my_extension",           # 编译后的模块名
    cpp_sources=cpp_code,          # C++ 源码
    cuda_sources=cuda_code,        # CUDA kernel 源码
    functions=["func1", "func2"],  # 要暴露的 C++ 函数名
    extra_cuda_cflags=["-O3", "-arch=sm_120"],
    extra_cflags=["-O3"],
    verbose=True                   # 显示编译过程
)

# 调用
result = module.func1(input_tensor)
```

### 3.2 优缺点

```
✅ 优点：
  - 不需要 CMake、setup.py
  - 快速迭代（改了源码重新 run 就行）
  - 适合学习和小型测试

❌ 缺点：
  - 不持久化（每次运行都重新编译）
  - 不适合多文件/复杂依赖
  - 难以调试
```

---

## 4. 方式二详解：`setup.py` + `JIT`（正式项目）

### 4.1 文件结构

```
rmsnorm/
├── setup.py
├── rmsnorm_cuda.cpp       # PyTorch binding
├── rmsnorm_kernel.cu      # CUDA kernel
└── test.py
```

### 4.2 `setup.py`

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rmsnorm_cuda",
    ext_modules=[
        CUDAExtension(
            name="rmsnorm_cuda",
            sources=[
                "rmsnorm_cuda.cpp",
                "rmsnorm_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-arch=sm_120"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

### 4.3 `rmsnorm_cuda.cpp`（PyTorch binding）

```cpp
// rmsnorm_cuda.cpp
#include <torch/extension.h>

// 声明 CUDA kernel launch 函数
torch::Tensor rmsnorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    float eps);

// 定义 Python 可见的接口
torch::Tensor rmsnorm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    float eps) {
    // 输入验证
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    return rmsnorm_forward_cuda(x, weight, eps);
}

// 注册为 PyTorch 内置风格的算子
TORCH_LIBRARY(my_rmsnorm, m) {
    m.def("rmsnorm(Tensor x, Tensor weight, float eps=1e-6) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_rmsnorm, CUDA, m) {
    m.impl("rmsnorm", rmsnorm_forward);
}

// Python 绑定（向后兼容方式）
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rmsnorm_forward, "RMSNorm forward");
}
```

### 4.4 安装和使用

```bash
# 安装
cd rmsnorm && pip install -e .

# 在 Python 中
import torch
import rmsnorm_cuda

x = torch.randn(4096, 4096, device="cuda")
w = torch.ones(4096, device="cuda")
y = rmsnorm_cuda.forward(x, w, 1e-6)

# 或用 TORCH_LIBRARY 注册的名字
y = torch.ops.my_rmsnorm.rmsnorm(x, w)
```

---

## 5. 方式三详解：`torch.library` + `autograd.Function`（生产级）

### 5.1 支持自动求导的完整算子

```python
# rmsnorm_autograd.py
import torch
from torch.autograd import Function

class RMSNormFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        """
        Forward pass: 调用你的 CUDA kernel
        """
        y = torch.ops.my_rmsnorm.rmsnorm(x, weight, eps)
        ctx.save_for_backward(x, weight, y)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: RMSNorm 的梯度计算
        """
        x, weight, y = ctx.saved_tensors
        eps = ctx.eps
        rows, cols = x.shape

        # RMSNorm 的梯度公式（简化版）：
        # dy/dx = weight / rms * (I - (1/cols) * y*y^T)
        rms = torch.sqrt((x * x).mean(dim=1, keepdim=True) + eps)

        grad_weight = (grad_output * x / rms).sum(dim=0)
        grad_x = weight / rms * (
            grad_output -
            (grad_output * y * weight / rms).sum(dim=1, keepdim=True) / cols * x / rms
        )

        return grad_x, grad_weight, None

# 使用 autograd.Function
def rmsnorm(x, weight, eps=1e-6):
    return RMSNormFunction.apply(x, weight, eps)

# 测试自动求导
x = torch.randn(4, 8, device="cuda", requires_grad=True)
w = torch.ones(8, device="cuda", requires_grad=True)
y = rmsnorm(x, w)
loss = y.sum()
loss.backward()
print("x.grad:", x.grad)
print("w.grad:", w.grad)
```

---

## 6. `nn.Module` 集成

```python
# rmsnorm_module.py
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        # x: [batch, ..., normalized_shape]
        # 假设最后一维是要 norm 的维度
        orig_shape = x.shape
        x_2d = x.view(-1, orig_shape[-1])
        y = RMSNormFunction.apply(x_2d, self.weight, self.eps)
        return y.view(orig_shape)

# 用法（像普通 PyTorch 模块一样！）
norm = RMSNorm(768).cuda()
x = torch.randn(32, 128, 768, device="cuda")
y = norm(x)  # 直接调用
```

---

## 7. PyTorch 中你必须了解的全部内容

### 7.1 核心概念地图

```
PyTorch 自定义算子开发 知识体系
│
├── 1. Tensor 内存模型（P0）
│   ├── data_ptr<T>() — 获取原始指针
│   ├── sizes() / strides() — 形状与步长
│   ├── is_contiguous() — 是否连续（不连续就不能传指针给 kernel）
│   └── .contiguous() — 显式连续化
│
├── 2. Dispatch 机制（P0）
│   ├── TORCH_LIBRARY — 注册算子定义
│   ├── TORCH_LIBRARY_IMPL — 注册特定后端实现
│   └── torch.ops.xxx.yyy — 调用注册的算子
│
├── 3. Python ↔ C++ 交互（P0）
│   ├── pybind11（PYBIND11_MODULE）
│   ├── torch::Tensor（C++ 端的张量类型）
│   └── .def("name", &func) — 暴露给 Python
│
├── 4. 自动求导（P1）
│   ├── torch.autograd.Function
│   ├── forward(ctx, *args) — 前向
│   ├── backward(ctx, *grads) — 反向
│   └── ctx.save_for_backward() — 保留反向需要的中间结果
│
├── 5. 设备与数据类型（P1）
│   ├── TORCH_CHECK(device().is_cuda()) — 设备检查
│   ├── torch::kCUDA / torch::kCPU
│   ├── torch::kFloat32 / torch::kFloat16 / torch::kBFloat16
│   └── options() — 创建张量的配置
│
├── 6. 性能集成（P1）
│   ├── torch.cuda.Stream — 流管理
│   ├── c10::cuda::getCurrentCUDAStream() — 获取当前流
│   └── AT_DISPATCH_FLOATING_TYPES — 类型分发宏
│
└── 7. 测试与部署（P2）
    ├── torch.testing.assert_close — 数值验证
    ├── torch.compile（TorchDynamo + Inductor）
    └── torch::deploy（C++ 推理服务）
```

### 7.2 关键细节

#### A. Tensor 连续性检查

```cpp
// ❌ 如果不检查，当用户传入转置后的 Tensor 时会出错
torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor weight, float eps) {
    // 检查连续性（自定义 CUDA kernel 通常要求连续输入）
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    // 如果不连续，应该先连续化
    x = x.contiguous();
    // ...
}
```

#### B. 获取当前 CUDA Stream

```cpp
#include <c10/cuda/CUDAStream.h>

// 获取当前 PyTorch 正在使用的 CUDA stream
// 确保你的 kernel launch 在正确的 stream 上
cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.device().index());
my_kernel<<<grid, block, smem, stream>>>(...);
```

#### C. 类型分发（支持 FP16/BF16/FP32）

```cpp
// 宏展开：根据 Tensor 的 dtype 调用不同的模板实例化
AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    x.scalar_type(),      // 输入 dtype
    "rmsnorm_forward",    // 错误信息前缀
    [&]() {
        // scalar_t 是当前类型的 C++ 类型
        using scalar_t = at::ScalarType;
        launch_kernel<scalar_t>(...);
    }
);
```

---

## 8. 从你的 CUDA 项目到 PyTorch 集成的完整清单

```
阶段 1：确保 kernel 可以独立工作
  ✅ 有独立的 .cu 文件 + main 函数
  ✅ 有正确性验证（CPU golden 对比）
  ✅ 有性能测试（cudaEvent timing）

阶段 2：写 PyTorch binding（~30 行代码）
  □ 创建 setup.py
  □ 创建 .cpp binding 文件（TORCH_LIBRARY 注册）
  □ 原 CUDA kernel 保持不变（或微调）

阶段 3：数值验证（在 PyTorch 中）
  □ torch.testing.assert_close(y, y_torch_ref)
  □ 测试非标准维度（奇数 cols, 大 batch）
  □ 测试不同 dtype（FP16/BF16）

阶段 4：求导支持
  □ 用 autograd.Function 包装
  □ torch.autograd.gradcheck() 验证梯度

阶段 5：生产优化
  □ 流管理（确保与 PyTorch 的其他 kernel 不冲突）
  □ 不同架构的编译选项（sm_80, sm_89, sm_90, sm_120）
  □ torch.compile 兼容性（TorchDynamo 是否能 trace 你的算子）
```

---

## 9. 常见坑与解决方案

| 问题 | 原因 | 解决 |
|------|------|------|
| `undefined symbol: cudaLaunchKernel` | 链接时缺少 CUDA runtime | 在 setup.py 中 `libraries=["cudart"]` |
| 结果全是 0 或 NaN | 共享内存大小设置错误 | 检查 `extern __shared__` 的大小计算 |
| CUDA kernel 和 PyTorch 结果不一致 | 数据类型/内存布局差异 | 检查 dtype、is_contiguous、strides |
| pip install 失败找不到 nvcc | CUDA toolkit 不在 PATH | `export PATH=/usr/local/cuda/bin:$PATH` |
| 编译通过但运行时 segmentation fault | block/grid dim 过大 | 检查矩阵维度和 block 大小 |
| `RuntimeError: CUDA error: invalid configuration argument` | block 线程数 > 1024 或 SMEM > 48KB | 减少 block size 或用 cudaFuncSetAttribute |

---

---
> 如果要支持训练，还会用 `torch.autograd.Function` 包装，自己写 backward
> 的梯度计算。最后用 `torch.autograd.gradcheck` 验证梯度正确性。
>
> 举个具体例子——我的 RMSNorm v3 就完成了完整的 PyTorch 集成，包括 weight
> 缓存优化和自动求导支持。"
