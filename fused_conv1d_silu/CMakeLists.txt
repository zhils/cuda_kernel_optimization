set(FUSED_CONV1D_SILU_VARIANTS
    fused_conv1d_silu_v0
)

foreach(variant ${FUSED_CONV1D_SILU_VARIANTS})
    add_executable(${variant} ${variant}.cu)
    target_link_libraries(${variant} PRIVATE cuda_common)
    set_target_properties(${variant} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()
