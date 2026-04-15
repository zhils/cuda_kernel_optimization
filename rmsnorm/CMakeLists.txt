set(RMSNORM_VARIANTS
    rmsnorm_v0_naive
    rmsnorm_v1
    rmsnorm_v2
    rmsnorm_v3
)

set(CUDNN_ROOT "C:/Program Files/NVIDIA/CUDNN/v9.20")
set(CUDNN_INCLUDE_DIR "${CUDNN_ROOT}/include/13.2")
set(CUDNN_LIB_DIR "${CUDNN_ROOT}/lib/13.2/x64")

foreach(variant ${RMSNORM_VARIANTS})
    add_executable(${variant} ${variant}.cu)
    target_link_libraries(${variant} PRIVATE cuda_common)
    set_target_properties(${variant} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()

add_executable(rmsnorm_benchmark_all benchmark_all.cu)
target_link_libraries(rmsnorm_benchmark_all PRIVATE cuda_common cudnn cublas)
target_include_directories(rmsnorm_benchmark_all PRIVATE "${CUDNN_INCLUDE_DIR}")
target_link_directories(rmsnorm_benchmark_all PRIVATE "${CUDNN_LIB_DIR}")
if(MSVC)
    target_compile_options(rmsnorm_benchmark_all PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()
set_target_properties(rmsnorm_benchmark_all PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
