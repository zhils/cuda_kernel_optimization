set(RMSNORM_VARIANTS
    rmsnorm_v0_naive
    rmsnorm_v1_warp_reduce
    rmsnorm_v2_vectorized
    rmsnorm_v3_fused
)

foreach(variant ${RMSNORM_VARIANTS})
    add_executable(${variant} ${variant}.cu)
    target_link_libraries(${variant} PRIVATE cuda_common)
    set_target_properties(${variant} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()

add_executable(rmsnorm_benchmark_all benchmark_all.cu)
target_link_libraries(rmsnorm_benchmark_all PRIVATE cuda_common cudnn)
set_target_properties(rmsnorm_benchmark_all PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
