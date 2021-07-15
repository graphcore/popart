include(EnableCompilerWarnings)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

add_popart_cpp_unit_test(
    inplaceaccumulategradpartialsintooptimizeraccumtensortests
    transform_unit_tests.cpp
    TEST_UTILS test-graphs-test-util
)
