set(SOFTMAX_VARIANTS
    softmax_v0_naive
    softmax_v1_shared_mem
    softmax_v2_online
    softmax_v3_warp_shuffle
)

foreach(variant ${SOFTMAX_VARIANTS})
    add_executable(${variant} ${variant}.cu)
    target_link_libraries(${variant} PRIVATE cuda_common)
    set_target_properties(${variant} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()

add_executable(softmax_cudnn_ref softmax_cudnn_ref.cu)
target_link_libraries(softmax_cudnn_ref PRIVATE cuda_common cudnn)
set_target_properties(softmax_cudnn_ref PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(softmax_benchmark_all benchmark_all.cu)
target_link_libraries(softmax_benchmark_all PRIVATE cuda_common cudnn)
set_target_properties(softmax_benchmark_all PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
