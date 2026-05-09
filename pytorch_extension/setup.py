#!/usr/bin/env python3
"""PyTorch custom CUDA operators for softmax and rmsnorm."""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda_kernel_ops",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="cuda_kernel_ops",
            sources=[
                "binding.cpp",
                "../softmax/softmax_kernels.cuh",
                "../rmsnorm/rmsnorm_kernels.cuh",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "-arch=sm_120",
                    "-gencode=arch=compute_120,code=sm_120",
                ],
            },
            include_dirs=[
                "..",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    description="PyTorch custom CUDA operators: softmax and rmsnorm",
)
