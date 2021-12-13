# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
add_popart_cpp_unit_test(unittest_willow_builder test_builder.cpp)
add_popart_cpp_unit_test(unittest_willow_error test_error.cpp)
add_popart_cpp_unit_test(unittest_willow_stochasticroundingassumptionverifier test_stochasticroundingassumptionverifier.cpp TEST_UTILS test-graphs-test-util)

add_subdirectory("op")
add_subdirectory("popx")
add_subdirectory("analysis")
