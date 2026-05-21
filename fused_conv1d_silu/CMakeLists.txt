add_library(fused_api STATIC
  src/fused_api.cpp
  src/fused_ref_cpu.cpp
  src/fused_gpu.cu
)
target_include_directories(fused_api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(fused_api PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fused_api PUBLIC cuda_common CUDA::cudart cublas cublasLt gemm_api)
target_include_directories(fused_api PRIVATE ${CMAKE_SOURCE_DIR}/gemm/include)
set_target_properties(fused_api PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

set(CUTLASS_DIR "$ENV{HOME}/cutlass" CACHE PATH "CUTLASS source root")
target_include_directories(fused_api PRIVATE "${CUTLASS_DIR}/include")
target_compile_options(fused_api PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

set(FUSED_CONV1D_SILU_VARIANTS
    fused_conv1d_silu_v0
    fused_conv1d_silu_v1
    fused_conv1d_silu_v2
    fused_conv1d_silu_v3
)

foreach(variant ${FUSED_CONV1D_SILU_VARIANTS})
    add_executable(${variant} ${variant}.cu)
    target_link_libraries(${variant} PRIVATE cuda_common)
    target_include_directories(${variant} PRIVATE "${CMAKE_SOURCE_DIR}")
    if(variant STREQUAL "fused_conv1d_silu_v3")
        set(CUTLASS_DIR "$ENV{HOME}/cutlass" CACHE PATH "CUTLASS source root")
        target_include_directories(${variant} PRIVATE "${CUTLASS_DIR}/include")
        target_compile_options(${variant} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        set_property(TARGET ${variant} PROPERTY CXX_STANDARD 17)
        set_property(TARGET ${variant} PROPERTY CXX_STANDARD_REQUIRED ON)
    endif()
    set_target_properties(${variant} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()

# Dtype benchmark — links against fused_api to reuse all dtype paths
add_executable(fused_conv1d_silu_dtype fused_conv1d_silu_dtype.cu)
target_link_libraries(fused_conv1d_silu_dtype PRIVATE cuda_common fused_api CUDA::cudart)
target_include_directories(fused_conv1d_silu_dtype PRIVATE "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(fused_conv1d_silu_dtype PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
