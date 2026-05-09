set(TEST_SOURCES
    test_runner.cpp
)

add_executable(unified_test ${TEST_SOURCES})
target_link_libraries(unified_test PRIVATE cuda_common)
set_target_properties(unified_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
