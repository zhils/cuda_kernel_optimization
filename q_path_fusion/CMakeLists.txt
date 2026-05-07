set(Q_PATH_FUSION_VARIANTS
    q_path_fusion_v0
    q_path_fusion_v1
    q_path_fusion_v2
)

foreach(variant ${Q_PATH_FUSION_VARIANTS})
    add_executable(${variant} ${variant}.cu)
    target_link_libraries(${variant} PRIVATE cuda_common)
    target_include_directories(${variant} PRIVATE "${CMAKE_SOURCE_DIR}")
    set_target_properties(${variant} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()

# Link cuBLAS for v2
target_link_libraries(q_path_fusion_v2 PRIVATE cublas)

