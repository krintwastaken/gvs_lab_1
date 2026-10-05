add_executable(run_benchmarks
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_bench.cu
)

target_link_libraries(run_benchmarks
    PRIVATE
        core_lib
        benchmark::benchmark
        Eigen3::Eigen
        CUDA::cudart
)

set_target_properties(run_benchmarks PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)