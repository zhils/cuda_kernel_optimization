cmake_minimum_required(VERSION 3.18)

add_executable(cko_unit_tests
  test_status.cpp
  test_catalog_loader.cpp
  test_gemm_validate.cpp
  test_gemm_ref_cpu.cpp
  test_gemm_gpu.cpp
  test_gemm_quant.cpp
  test_rmsnorm_validate.cpp
  test_rmsnorm_ref_cpu.cpp
  test_rmsnorm_gpu.cpp
  test_fused_validate.cpp
  test_fused_ref_cpu.cpp
  test_fused_gpu.cpp
  test_c_abi.cpp
)

target_compile_definitions(cko_unit_tests PRIVATE CKO_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

target_include_directories(cko_unit_tests PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(cko_unit_tests PRIVATE
  gemm_api
  rmsnorm_api
  fused_api
  GTest::gtest_main
  CUDA::cudart
  cublas
  cublasLt
)

include(GoogleTest)
gtest_discover_tests(cko_unit_tests)
