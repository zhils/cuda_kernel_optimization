cmake_minimum_required(VERSION 3.18)

add_library(cuda_common STATIC
  src/benchmark.cpp
  src/test_case_loader.cpp
)

target_include_directories(cuda_common
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(cuda_common
  PUBLIC
    nlohmann_json::nlohmann_json
)

target_compile_features(cuda_common PUBLIC cxx_std_17)
