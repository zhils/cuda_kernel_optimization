set(GEMM_VARIANTS
    gemm_v0
    gemm_v1
    gemm_v2
    gemm_v3
    gemm_v4
    gemm_cublas_ref
)

set(GEMM_VARIANTS_ASM
    gemm_v0
    gemm_v1
    gemm_v2
    gemm_v3
    gemm_v4
)

foreach(variant ${GEMM_VARIANTS})
    add_executable(${variant} ${variant}.cu)
    target_link_libraries(${variant} PRIVATE cuda_common)
    set_target_properties(${variant} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()

target_link_libraries(gemm_cublas_ref PRIVATE cublas)

set(ASM_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/gemm/asm)
set(NVCC_FLAGS -Xcompiler=-Wno-deprecated-declarations)

foreach(variant IN LISTS GEMM_VARIANTS)
    if(${variant} STREQUAL "gemm_cublas_ref")
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
    gemm_v0 gemm_v1 gemm_v2 gemm_v3 gemm_v4
)

add_custom_target(gen_asm_sass ALL DEPENDS
    gemm_v0 gemm_v1 gemm_v2 gemm_v3 gemm_v4
)

