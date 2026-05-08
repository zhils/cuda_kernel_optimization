set(FLASH_ATTENTION_VARIANTS
    flash_attention_v0
    flash_attention_v1
    flash_attention_v2
)

foreach(variant ${FLASH_ATTENTION_VARIANTS})
    add_executable(${variant} ${variant}.cu)
    target_link_libraries(${variant} PRIVATE cuda_common)
    set_target_properties(${variant} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()
