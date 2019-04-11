# Tests of ONNX If Operator
# to run only these tests with ctest, consider using something like,
# `ctest -R logical_if_test -j5 -VV'.

add_poponnx_cpp_unit_test(builder_0_logical_if_test builder_0_logical_if_test.cpp)
add_poponnx_cpp_unit_test(basic_0_logical_if_test basic_0_logical_if_test.cpp)

# This test requires scopes to appear in Ir Tensor identification
# add_poponnx_cpp_unit_test(namescope_0_logical_if_test namescope_0_logical_if_test.cpp)
