cmake_minimum_required(VERSION 3.18)

add_library(cuda_common STATIC
  src/benchmark.cpp
)

target_include_directories(cuda_common
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(cuda_common PUBLIC cxx_std_17)
