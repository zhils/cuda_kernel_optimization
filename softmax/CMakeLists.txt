set(SOFTMAX_VARIANTS
    softmax_v0
    softmax_v1
    softmax_v2
    softmax_v3
)

set(CUDNN_ROOT "C:/Program Files/NVIDIA/CUDNN/v9.20")
set(CUDNN_INCLUDE_DIR "${CUDNN_ROOT}/include/13.2")
set(CUDNN_LIB_DIR "${CUDNN_ROOT}/lib/13.2/x64")

foreach(variant ${SOFTMAX_VARIANTS})
    add_executable(${variant} ${variant}.cu)
    target_link_libraries(${variant} PRIVATE cuda_common)
    set_target_properties(${variant} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_ARCHITECTURES "120")
endforeach()

add_executable(softmax_cudnn_ref softmax_cudnn_ref.cu)
target_link_libraries(softmax_cudnn_ref PRIVATE cuda_common cudnn)
target_include_directories(softmax_cudnn_ref PRIVATE "${CUDNN_INCLUDE_DIR}")
target_link_directories(softmax_cudnn_ref PRIVATE "${CUDNN_LIB_DIR}")
set_target_properties(softmax_cudnn_ref PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(softmax_benchmark_all benchmark_all.cu)
target_link_libraries(softmax_benchmark_all PRIVATE cuda_common cudnn cublas)
target_include_directories(softmax_benchmark_all PRIVATE "${CUDNN_INCLUDE_DIR}")
target_link_directories(softmax_benchmark_all PRIVATE "${CUDNN_LIB_DIR}")
if(MSVC)
    target_compile_options(softmax_benchmark_all PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()
set_target_properties(softmax_benchmark_all PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


