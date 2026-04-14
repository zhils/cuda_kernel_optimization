# Benchmark Environment

## Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB |
| Architecture | Blackwell (sm_120, Compute Capability 12.0) |
| SMs | 36 |
| CUDA Cores | 4608 |
| Tensor Cores | 4th gen |
| VRAM | 16 GB GDDR7 |
| Memory Bus | 128-bit |
| Memory Bandwidth | ~XX GB/s |
| L2 Cache | 48 MB |
| FP32 Peak | ~XX TFLOPS |
| TDP | 150W |

## Software

| Component | Version |
|-----------|---------|
| CUDA Toolkit | 12.9 |
| cuBLAS | Bundled with CUDA 12.9 |
| cuDNN | Latest |
| NVIDIA Driver | 550.90+ |
| nvcc | 12.9 |
| CMake | 3.18+ |
| OS | Windows 11 / Linux |

## Benchmark Methodology

1. **Warmup:** 3 iterations before timed runs (flush instruction/data caches, JIT compilation)
2. **Timing:** `cudaEvent` based (GPU-side, no CPU overhead)
3. **Iterations:** 10 timed runs per configuration
4. **Metric:** Trimmed mean (discard min and max, average remaining 8)
5. **Correctness:** Every kernel output is verified against CPU reference (`max_abs_diff < threshold`)
6. **Throttling:** Ensure GPU is at base clock (no thermal throttling) via `nvidia-smi`

## How to Reproduce

```bash
# Verify GPU status
nvidia-smi

# Build
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=120
make -j$(nproc)

# Run benchmarks from project root
cd ..
./build/bin/gemm_benchmark_all
./build/bin/softmax_benchmark_all
./build/bin/rmsnorm_benchmark_all
```
