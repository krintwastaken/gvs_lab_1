add_library(core_lib INTERFACE)

target_include_directories(core_lib
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(core_lib
    INTERFACE
        CUDA::cudart
)