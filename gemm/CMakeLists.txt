set(GEMM_VARIANTS
    gemm_v0_naive
    gemm_v1_tiled_smem
    gemm_v2_thread_tiling
    gemm_v3_fastpath
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
