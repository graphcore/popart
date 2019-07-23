# Tests of pipelining
# to run only these tests with ctest, consider using something like,
# `ctest -R pipelin -j5 -VV'.

add_poponnx_cpp_unit_test(pipeline_topo_con_test_0 pipeline_topo_con_test_0.cpp)
add_poponnx_cpp_unit_test(pipeline_continuous_equivalent_test_0 pipeline_continuous_equivalent_test_0.cpp)
