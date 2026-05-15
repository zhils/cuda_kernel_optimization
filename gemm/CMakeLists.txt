set(GEMM_VARIANTS
    gemm_v0
    gemm_v1
    gemm_v2
    gemm_v3
    gemm_v4
    gemm_fp16
    gemm_bf16
    gemm_fp8
    gemm_int8
    gemm_fp8_cublaslt
    quant_gemm_compare
    quant_fp8_bias_compensate
    gemm_cublas_ref
)

set(GEMM_VARIANTS_ASM
    gemm_v0
    gemm_v1
    gemm_v2
    gemm_v3
    gemm_v4
    gemm_bf16
)

foreach(variant ${GEMM_VARIANTS})
    add_executable(${variant} ${variant}.cu)
    target_link_libraries(${variant} PRIVATE cuda_common)
    set_target_properties(${variant} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()

# CUDA 13.2 + sm_120 toolchain does not expose tf32 WMMA tags by default.
# Enable Ampere WMMA tag path for legacy tf32 kernels.
target_compile_definitions(gemm_v4 PRIVATE __CUDA_AMPERE_MMA__)

target_link_libraries(gemm_cublas_ref PRIVATE cublas)
target_link_libraries(gemm_fp8 PRIVATE cublasLt)
target_link_libraries(gemm_fp8_cublaslt PRIVATE cublas cublasLt)

# gemm_all_compare: unified GEMM comparison benchmark
set(GEMM_ALL_COMPARE_SRCS gemm_all_compare.cu)
add_executable(gemm_all_compare ${GEMM_ALL_COMPARE_SRCS})
target_link_libraries(gemm_all_compare PRIVATE cuda_common cublas)
target_compile_definitions(gemm_all_compare PRIVATE ALL_COMPARE_LIB)
if(DEFINED CUTLASS_ROOT)
    get_filename_component(CUTLASS_ABS_ROOT "${CUTLASS_ROOT}" ABSOLUTE)
    message(STATUS "CUTLASS_ROOT = ${CUTLASS_ABS_ROOT}")
    target_include_directories(gemm_all_compare PRIVATE ${CUTLASS_ABS_ROOT}/include)
    target_compile_definitions(gemm_all_compare PRIVATE CUTLASS_ROOT)
endif()
set_target_properties(gemm_all_compare PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(ASM_OUTPUT_DIR ${CMAKE_BINARY_DIR}/gemm/asm)
set(NVCC_FLAGS -Xcompiler=-Wno-deprecated-declarations)

foreach(variant IN LISTS GEMM_VARIANTS)
    if(${variant} STREQUAL "gemm_cublas_ref")
        continue()
    endif()
    if(${variant} STREQUAL "quant_gemm_compare")
        continue()
    endif()
    if(${variant} STREQUAL "quant_fp8_bias_compensate")
        continue()
    endif()
    if(${variant} STREQUAL "gemm_fp8_cublaslt")
        continue()
    endif()
    add_custom_command(
        OUTPUT ${ASM_OUTPUT_DIR}/ptx/${variant}.ptx
        COMMAND ${CMAKE_CUDA_COMPILER} ${NVCC_FLAGS} -ptx
                ${CMAKE_SOURCE_DIR}/gemm/${variant}.cu
                --generate-code=arch=compute_89,code=sm_89
                -o ${ASM_OUTPUT_DIR}/ptx/${variant}.ptx
                -I${CMAKE_SOURCE_DIR}/common/include
                -I${CMAKE_CUDA_TOOLKIT_INCLUDE}
        DEPENDS ${variant}
        COMMENT "Generating PTX for ${variant}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${ASM_OUTPUT_DIR}/sass/${variant}.cubin
        COMMAND ${CMAKE_CUDA_COMPILER} ${NVCC_FLAGS} -cubin
                ${CMAKE_SOURCE_DIR}/gemm/${variant}.cu
                --generate-code=arch=compute_89,code=sm_89
                -o ${ASM_OUTPUT_DIR}/sass/${variant}.cubin
                -I${CMAKE_SOURCE_DIR}/common/include
                -I${CMAKE_CUDA_TOOLKIT_INCLUDE}
        DEPENDS ${variant}
        COMMENT "Generating SASS (cubin) for ${variant}"
        VERBATIM
    )
endforeach()

add_custom_target(gen_asm_ptx ALL DEPENDS
    gemm_v0 gemm_v1 gemm_v2 gemm_v3 gemm_v4 gemm_bf16
)

add_custom_target(gen_asm_sass ALL DEPENDS
    gemm_v0 gemm_v1 gemm_v2 gemm_v3 gemm_v4 gemm_bf16
)
