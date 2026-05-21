add_library(gemm_api STATIC
  src/gemm_api.cpp
  src/gemm_ref_cpu.cpp
  src/gemm_gpu.cu
  src/gemm_quant.cu
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(gemm_api PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(gemm_api PRIVATE $<$<COMPILE_LANGUAGE:CXX>:_OPENMP>)
endif()

target_include_directories(gemm_api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(gemm_api PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
target_link_libraries(gemm_api PUBLIC cuda_common CUDA::cudart cublas cublasLt)
set_target_properties(gemm_api PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

set(CUTLASS_DIR "$ENV{HOME}/cutlass" CACHE PATH "CUTLASS source root")

if(EXISTS "${CUTLASS_DIR}/include/cute/tensor.hpp")
  set(CUTLASS_UTIL_INCLUDE "${CUTLASS_DIR}/tools/util/include")
  set(CUTLASS_GEMM_CUDA_FLAGS "--expt-relaxed-constexpr")
  target_include_directories(gemm_api PRIVATE "${CUTLASS_DIR}/include" "${CUTLASS_UTIL_INCLUDE}")
  target_compile_definitions(gemm_api PRIVATE CKO_GEMM_CUTLASS_FP16)
  target_compile_options(gemm_api PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:${CUTLASS_GEMM_CUDA_FLAGS}>)
  message(STATUS "gemm: API kFp16 uses CUTLASS (${CUTLASS_DIR})")
  message(STATUS "gemm: CuTe FP16 binary enabled (${CUTLASS_DIR})")
else()
  message(STATUS "gemm: API kFp16 uses handwritten WMMA (CUTLASS not found at ${CUTLASS_DIR})")
  message(STATUS "gemm: CuTe FP16 binary disabled (CUTLASS not found at ${CUTLASS_DIR})")
endif()

set(GEMM_VARIANTS
    gemm_v0
    gemm_v1
    gemm_v2
    gemm_v3
    gemm_v4
    gemm_fp16
    gemm_cublas_ref
    gemm_cublas_fp16
)

foreach(variant ${GEMM_VARIANTS})
    add_executable(${variant} ${variant}.cu)
    target_link_libraries(${variant} PRIVATE cuda_common)
    set_target_properties(${variant} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()

if(EXISTS "${CUTLASS_DIR}/include/cute/tensor.hpp")
  add_executable(gemm_cute_fp16 gemm_cute_fp16.cu)
  target_link_libraries(gemm_cute_fp16 PRIVATE cuda_common)
  target_include_directories(gemm_cute_fp16 PRIVATE "${CMAKE_SOURCE_DIR}" "${CUTLASS_DIR}/include"
      "${CUTLASS_UTIL_INCLUDE}")
  target_compile_options(gemm_cute_fp16 PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:${CUTLASS_GEMM_CUDA_FLAGS}>)
  set_target_properties(gemm_cute_fp16 PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

target_link_libraries(gemm_cublas_ref PRIVATE cublas)
target_link_libraries(gemm_cublas_fp16 PRIVATE cublas)

add_executable(gemm_dtype gemm_dtype.cu)
target_link_libraries(gemm_dtype PRIVATE gemm_api cuda_common CUDA::cudart cublas cublasLt)
target_include_directories(gemm_dtype PRIVATE "${CMAKE_SOURCE_DIR}")
set_target_properties(gemm_dtype PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(gemm_tune gemm_tune.cu)
target_link_libraries(gemm_tune PRIVATE gemm_api cuda_common CUDA::cudart)
target_include_directories(gemm_tune PRIVATE "${CMAKE_SOURCE_DIR}")
set_target_properties(gemm_tune PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(gemm_quant_bench gemm_quant_bench.cu)
target_link_libraries(gemm_quant_bench PRIVATE gemm_api cuda_common CUDA::cudart)
target_include_directories(gemm_quant_bench PRIVATE "${CMAKE_SOURCE_DIR}")
set_target_properties(gemm_quant_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(gemm_compare gemm_compare.cu)
target_link_libraries(gemm_compare PRIVATE gemm_api cuda_common CUDA::cudart cublas cublasLt)
target_include_directories(gemm_compare PRIVATE "${CMAKE_SOURCE_DIR}")
set_target_properties(gemm_compare PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
