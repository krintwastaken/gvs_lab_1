add_executable(run_benchmarks
    src/main_bench.cpp
)

target_link_libraries(run_benchmarks
    PRIVATE
        core_lib
        benchmark::benchmark
        Eigen3::Eigen
)