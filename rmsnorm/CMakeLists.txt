add_library(rmsnorm_api STATIC
  src/rmsnorm_api.cpp
  src/rmsnorm_ref_cpu.cpp
  src/rmsnorm_gpu.cu
)
target_include_directories(rmsnorm_api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(rmsnorm_api PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
target_link_libraries(rmsnorm_api PUBLIC cuda_common CUDA::cudart)
set_target_properties(rmsnorm_api PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

set(RMSNORM_VARIANTS
    rmsnorm_v0
    rmsnorm_v1
    rmsnorm_v2
    rmsnorm_v3
    rmsnorm_cub_ref
)

foreach(variant ${RMSNORM_VARIANTS})
    add_executable(${variant} ${variant}.cu)
    target_link_libraries(${variant} PRIVATE cuda_common)
    target_include_directories(${variant} PRIVATE "${CMAKE_SOURCE_DIR}")
    set_target_properties(${variant} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()

add_executable(rmsnorm_dtype rmsnorm_dtype.cu)
target_link_libraries(rmsnorm_dtype PRIVATE rmsnorm_api cuda_common CUDA::cudart)
target_include_directories(rmsnorm_dtype PRIVATE "${CMAKE_SOURCE_DIR}")
set_target_properties(rmsnorm_dtype PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(rmsnorm_compare rmsnorm_compare.cu)
target_link_libraries(rmsnorm_compare PRIVATE rmsnorm_api cuda_common CUDA::cudart)
target_include_directories(rmsnorm_compare PRIVATE "${CMAKE_SOURCE_DIR}")
set_target_properties(rmsnorm_compare PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

