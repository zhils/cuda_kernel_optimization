cmake_minimum_required(VERSION 3.18)
project(cuda_kernel_optimization CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_subdirectory(common)
add_subdirectory(gemm)
add_subdirectory(softmax)
add_subdirectory(rmsnorm)
add_subdirectory(q_path_fusion)
add_subdirectory(fused_conv1d_silu)
add_subdirectory(flash_attention)
add_subdirectory(fused_l2_norm_qk)
add_subdirectory(fused_gated_delta_rule)
add_subdirectory(fused_output_norm_gate)

# 统一测试框架（可选，默认不构建）
option(BUILD_TESTS "Build unified test framework" OFF)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
