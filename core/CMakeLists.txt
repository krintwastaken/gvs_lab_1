add_library(core_lib
    ${CMAKE_CURRENT_SOURCE_DIR}/src/VectorOps.cu
)

target_include_directories(core_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(core_lib
    PUBLIC
        CUDA::cudart
)

set_target_properties(core_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)