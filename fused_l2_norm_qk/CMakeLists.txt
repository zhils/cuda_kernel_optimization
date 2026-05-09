set(FUSED_L2_NORM_QK_VARIANTS
    fused_l2_norm_qk_v0
    fused_l2_norm_qk_v1
    fused_l2_norm_qk_v2
)

foreach(variant ${FUSED_L2_NORM_QK_VARIANTS})
    add_executable(${variant} ${variant}.cu)
    target_link_libraries(${variant} PRIVATE cuda_common)
    set_target_properties(${variant} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()
