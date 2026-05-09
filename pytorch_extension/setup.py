#!/usr/bin/env python3
"""PyTorch custom CUDA operators for softmax and rmsnorm."""

import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = [
    "binding.cpp",
    os.path.join("..", "softmax", "softmax_v0.cu"),
    os.path.join("..", "softmax", "softmax_v1.cu"),
    os.path.join("..", "softmax", "softmax_v2.cu"),
    os.path.join("..", "softmax", "softmax_v3.cu"),
    os.path.join("..", "rmsnorm", "rmsnorm_v0.cu"),
    os.path.join("..", "rmsnorm", "rmsnorm_v1.cu"),
    os.path.join("..", "rmsnorm", "rmsnorm_v2.cu"),
    os.path.join("..", "rmsnorm", "rmsnorm_v3.cu"),
]

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
    ],
}

arches = os.environ.get("TORCH_CUDA_ARCH_LIST", "120").split(";")
for arch in arches:
    arch = arch.strip()
    if arch:
        extra_compile_args["nvcc"].extend([
            f"-arch=sm_{arch}",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
        ])

include_dirs = [
    os.path.join("..", "common", "include"),
    os.path.join("..", "softmax"),
    os.path.join("..", "rmsnorm"),
]

setup(
    name="cuda_kernel_ops",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="cuda_kernel_ops",
            sources=sources,
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    description="PyTorch custom CUDA operators: softmax and rmsnorm",
)
