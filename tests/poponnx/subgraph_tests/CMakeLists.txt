# Tests of sub-graphs using our sub-string algorithm
# to run only these tests with ctest, consider using something like,
# `ctest -R subgraph_test -j5 -VV'.

# repeated sub-string functionality (pure string matching)
add_poponnx_cpp_unit_test(substring_0_subgraph_test 
   substring_0_subgraph_test.cpp)

# isomorphic sub-string functionality ("chars" are nodes)
add_poponnx_cpp_unit_test(isomorphism_0_subgraph_test 
   isomorphism_0_subgraph_test.cpp)

# removal of overlapping sub-string functionality
add_poponnx_cpp_unit_test(overlapping_0_subgraph_test 
   overlapping_0_subgraph_test.cpp)

# setting the value of a sub-graph, test of functionality
add_poponnx_cpp_unit_test(value_0_subgraph_test 
   value_0_subgraph_test.cpp)

# getting the final sub-graphs, testing functionality
add_poponnx_cpp_unit_test(final_0_subgraph_test 
   final_0_subgraph_test.cpp)

# tests of full functionality (combines all of the above)
add_poponnx_cpp_unit_test(getmatches_0_subgraph_test 
   getmatches_0_subgraph_test.cpp)

# a speed test (in development, see T7258)
add_poponnx_cpp_unit_test(speed_0_subgraph_test speed_0_subgraph_test.cpp)
