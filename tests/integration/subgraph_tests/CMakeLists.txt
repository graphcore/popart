# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Tests of sub-graphs using our sub-string algorithm
# to run only these tests with ctest, consider using something like,
# `ctest -R subgraph_test -j5 -VV'.

# tests of the Match class
add_unit_test(match_class_test
    match_class_test.cpp)

# repeated sub-string functionality (pure string matching)
add_unit_test(substring_0_subgraph_test
   substring_0_subgraph_test.cpp)

# isomorphic sub-string functionality ("chars" are nodes)
add_unit_test(isomorphism_0_subgraph_test
   isomorphism_0_subgraph_test.cpp)

# removal of overlapping sub-string functionality
add_unit_test(overlapping_0_subgraph_test
   overlapping_0_subgraph_test.cpp)

# setting the value of a sub-graph, test of functionality
add_unit_test(value_0_subgraph_test
   value_0_subgraph_test.cpp)

# getting the final sub-graphs, testing functionality
add_unit_test(final_0_subgraph_test
   final_0_subgraph_test.cpp)

# tests of full functionality (combines all of the above)
add_unit_test(getmatches_0_subgraph_test
   getmatches_0_subgraph_test.cpp)

# a speed test
add_unit_test(speed_0_subgraph_test speed_0_subgraph_test.cpp)

# tests at the Op level
add_unit_test(op_0_subgraph_test op_0_subgraph_test.cpp)

# tests at the Op level
add_unit_test(introspection_0_subgraph_test
	introspection_0_subgraph_test.cpp)

add_unit_test(slice_outlining_0_numerical_test
  slice_outlining_0_numerical_test.cpp)

#
add_unit_test(prune_matches_0_subgraph_test
	prune_matches_0_subgraph_test.cpp)

add_popart_py_unit_test(verify_subgraphs_test)

add_popart_py_unit_test(subgraph_regression_test)
