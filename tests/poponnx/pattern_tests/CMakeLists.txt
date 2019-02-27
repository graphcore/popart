# Tests of inplace / view-changing logic
# to run only these tests with ctest, consider using something like,
# `ctest -R ip_test -j5 -VV'.

add_poponnx_cpp_unit_test(softmaxgraddirect_0_patterns_test 
                          softmaxgraddirect_0_patterns_test.cpp)

add_poponnx_cpp_unit_test(patterns_test patterns_test.cpp)
