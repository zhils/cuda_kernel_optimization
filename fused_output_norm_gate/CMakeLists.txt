set(FUSED_OUTPUT_NORM_GATE_VARIANTS
    fused_output_norm_gate_v0
    fused_output_norm_gate_v1
    fused_output_norm_gate_v2
)

foreach(variant ${FUSED_OUTPUT_NORM_GATE_VARIANTS})
    add_executable(${variant} ${variant}.cu)
    target_link_libraries(${variant} PRIVATE cuda_common)
    set_target_properties(${variant} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()
