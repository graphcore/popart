# Tests of ONNX TopK Operator
# to run only these tests with ctest, consider using something like,
# `ctest -R topk -j5 -VV'.

add_poponnx_cpp_unit_test(basic0_topk_test basic0_topk_test.cpp)
