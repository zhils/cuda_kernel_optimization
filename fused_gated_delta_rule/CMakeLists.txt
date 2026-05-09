set(FUSED_GATED_DELTA_RULE_VARIANTS
    fused_gated_delta_rule_v0
    fused_gated_delta_rule_v1
    fused_gated_delta_rule_v2
    fused_gated_delta_rule_v3
)

foreach(variant ${FUSED_GATED_DELTA_RULE_VARIANTS})
    add_executable(${variant} ${variant}.cu)
    target_link_libraries(${variant} PRIVATE cuda_common)
    set_target_properties(${variant} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()
