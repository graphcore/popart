# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
add_unit_test(unittest_willow_builder test_builder.cpp)
add_unit_test(unittest_willow_commgroup test_commgroup.cpp)
add_unit_test(unittest_willow_error test_error.cpp)
add_unit_test(unittest_willow_replicagrouping test_replicagrouping.cpp)
add_unit_test(unittest_willow_stochasticroundingassumptionverifier test_stochasticroundingassumptionverifier.cpp SUPPORT_LIBS test-graphs-test-util)
add_unit_test(unittest_willow_tensornames test_tensornames.cpp)
add_unit_test(unittest_willow_variablesettings test_variablesettings.cpp)
add_unit_test(unittest_willow_variablesettingsdomain test_variablesettingsdomain.cpp)

add_subdirectory("op")
add_subdirectory("popx")
add_subdirectory("analysis")
