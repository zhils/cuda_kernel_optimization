set(GEMM_VARIANTS
    gemm_v0_naive
    gemm_v1
    gemm_v2
    gemm_v3
    gemm_v4
    gemm_cublas_ref
)

foreach(variant ${GEMM_VARIANTS})
    add_executable(${variant} ${variant}.cu)
    target_link_libraries(${variant} PRIVATE cuda_common)
    set_target_properties(${variant} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()

target_link_libraries(gemm_cublas_ref PRIVATE cublas)

add_executable(gemm_benchmark_all benchmark_all.cu)
target_link_libraries(gemm_benchmark_all PRIVATE cuda_common cublas)
set_target_properties(gemm_benchmark_all PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(gemm_benchmark_v2_blocks benchmark_v2_blocks.cu)
target_link_libraries(gemm_benchmark_v2_blocks PRIVATE cuda_common cublas)
set_target_properties(gemm_benchmark_v2_blocks PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(gemm_benchmark_v2v3v4 benchmark_v2v3v4.cu)
target_link_libraries(gemm_benchmark_v2v3v4 PRIVATE cuda_common)
set_target_properties(gemm_benchmark_v2v3v4 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
