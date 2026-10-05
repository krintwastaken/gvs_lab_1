add_executable(run_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_test.cu
)

target_link_libraries(run_tests
    PRIVATE
        core_lib
        GTest::gtest
        GTest::gtest_main
        Eigen3::Eigen
        CUDA::cudart
)

set_target_properties(run_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

include(GoogleTest)
gtest_discover_tests(run_tests)