add_executable(run_tests
    src/main_test.cpp
)

target_link_libraries(run_tests
    PRIVATE
        core_lib
        GTest::gtest
        GTest::gtest_main
        Eigen3::Eigen
)

include(GoogleTest)
gtest_discover_tests(run_tests)