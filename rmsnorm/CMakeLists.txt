set(RMSNORM_VARIANTS
    rmsnorm_v0
    rmsnorm_v1
    rmsnorm_v2
    rmsnorm_v3
    rmsnorm_cub_ref
)

foreach(variant ${RMSNORM_VARIANTS})
    add_executable(${variant} ${variant}.cu)
    target_link_libraries(${variant} PRIVATE cuda_common)
    target_include_directories(${variant} PRIVATE "${CMAKE_SOURCE_DIR}")
    set_target_properties(${variant} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()

