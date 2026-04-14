cmake_minimum_required(VERSION 3.18)
project(cuda_kernel_optimization CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_subdirectory(common)
add_subdirectory(gemm)
add_subdirectory(softmax)
add_subdirectory(rmsnorm)
