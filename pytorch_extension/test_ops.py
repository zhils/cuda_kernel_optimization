#!/usr/bin/env python3
"""Test script for custom CUDA operators."""

import torch
import torch.nn.functional as F

# Import the compiled extension
try:
    import cuda_kernel_ops
except ImportError:
    print("Extension not compiled. Run: cd pytorch_extension && pip install -e .")
    raise


def test_softmax():
    """Test softmax against PyTorch reference."""
    print("=" * 60)
    print("Testing Softmax")
    print("=" * 60)

    rows, cols = 128, 256
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)

    # PyTorch reference
    y_ref = F.softmax(x, dim=1)

    for version in [0, 1, 2, 3]:
        y = cuda_kernel_ops.softmax_forward(x, version)
        max_diff = (y - y_ref).abs().max().item()
        print(f"  Version {version}: max_diff = {max_diff:.6e}", end="")
        print("  PASS" if max_diff < 1e-3 else "  FAIL")

    print()


def test_rmsnorm():
    """Test rmsnorm against manual implementation."""
    print("=" * 60)
    print("Testing RMSNorm")
    print("=" * 60)

    rows, cols = 128, 256
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    weight = torch.ones(cols, device="cuda", dtype=torch.float32)
    eps = 1e-5

    # Manual reference
    rms = torch.sqrt((x * x).mean(dim=1, keepdim=True) + eps)
    y_ref = x / rms * weight

    for version in [0, 1, 2, 3]:
        y = cuda_kernel_ops.rmsnorm_forward(x, weight, eps, version)
        max_diff = (y - y_ref).abs().max().item()
        print(f"  Version {version}: max_diff = {max_diff:.6e}", end="")
        print("  PASS" if max_diff < 1e-3 else "  FAIL")

    print()


def test_torch_ops_api():
    """Test torch.ops style API."""
    print("=" * 60)
    print("Testing torch.ops API")
    print("=" * 60)

    rows, cols = 64, 128
    x = torch.randn(rows, cols, device="cuda")
    weight = torch.ones(cols, device="cuda")

    # Softmax
    y1 = torch.ops.cuda_kernel_ops.softmax_forward(x, 3)
    y_ref = F.softmax(x, dim=1)
    print(f"  softmax: max_diff = {(y1 - y_ref).abs().max().item():.6e}")

    # RMSNorm
    y2 = torch.ops.cuda_kernel_ops.rmsnorm_forward(x, weight, 1e-5, 3)
    rms = torch.sqrt((x * x).mean(dim=1, keepdim=True) + 1e-5)
    y_ref2 = x / rms * weight
    print(f"  rmsnorm: max_diff = {(y2 - y_ref2).abs().max().item():.6e}")

    print()


def benchmark():
    """Simple benchmark."""
    print("=" * 60)
    print("Benchmark")
    print("=" * 60)

    rows, cols = 4096, 4096
    x = torch.randn(rows, cols, device="cuda")
    weight = torch.ones(cols, device="cuda")

    # Warmup
    for _ in range(10):
        cuda_kernel_ops.softmax_forward(x, 3)
        cuda_kernel_ops.rmsnorm_forward(x, weight, 1e-5, 3)
    torch.cuda.synchronize()

    # Benchmark softmax
    import time
    n_iters = 100

    start = time.perf_counter()
    for _ in range(n_iters):
        cuda_kernel_ops.softmax_forward(x, 3)
    torch.cuda.synchronize()
    softmax_ms = (time.perf_counter() - start) / n_iters * 1000
    print(f"  Softmax V3: {softmax_ms:.3f} ms")

    # Benchmark rmsnorm
    start = time.perf_counter()
    for _ in range(n_iters):
        cuda_kernel_ops.rmsnorm_forward(x, weight, 1e-5, 3)
    torch.cuda.synchronize()
    rmsnorm_ms = (time.perf_counter() - start) / n_iters * 1000
    print(f"  RMSNorm V3: {rmsnorm_ms:.3f} ms")

    print()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests.")
        exit(0)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    test_softmax()
    test_rmsnorm()
    test_torch_ops_api()
    benchmark()

    print("All tests completed!")
