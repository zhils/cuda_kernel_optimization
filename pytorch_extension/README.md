# PyTorch Custom CUDA Operators

PyTorch extension for custom CUDA kernels: **Softmax** and **RMSNorm**.

## File Structure

```
pytorch_extension/
├── setup.py          # Build configuration
├── binding.cpp       # C++ binding (PyTorch ↔ CUDA)
└── test_ops.py       # Python test script
```

## Build

```bash
cd pytorch_extension
pip install -e .
```

## Usage

### Python API

```python
import torch
import cuda_kernel_ops

# Softmax
x = torch.randn(4096, 4096, device="cuda")
y = cuda_kernel_ops.softmax_forward(x, version=3)

# RMSNorm
x = torch.randn(4096, 4096, device="cuda")
weight = torch.ones(4096, device="cuda")
y = cuda_kernel_ops.rmsnorm_forward(x, weight, eps=1e-5, version=3)
```

### torch.ops API

```python
y = torch.ops.cuda_kernel_ops.softmax_forward(x, 3)
y = torch.ops.cuda_kernel_ops.rmsnorm_forward(x, weight, 1e-5, 3)
```

## Version Selection

| Version | Softmax | RMSNorm |
|---------|---------|---------|
| 0 | Naive (1 thread/row) | Naive (1 thread/row) |
| 1 | Warp/row, serial reduce | Warp/row, serial reduce |
| 2 | Warp/row, shuffle reduce | Warp/row, shuffle reduce |
| 3 | Online softmax | Weight in SMEM |

## Test

```bash
python test_ops.py
```

## Known Gaps And Next Steps

- Current extension focuses on forward APIs; backward/autograd kernels are not included.
- Recommended additions: version compatibility matrix (PyTorch/CUDA), and benchmark parity checks against `build/bin` executables.
