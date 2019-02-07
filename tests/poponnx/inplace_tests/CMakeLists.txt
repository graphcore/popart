# Tests of inplace / view-changing logic
# to run only these tests with ctest, consider using something like,
# `ctest -R ip_test -j5 -VV'.

add_poponnx_cpp_unit_test(parallel_0_ip_test parallel_0_ip_test.cpp)
add_poponnx_cpp_unit_test(basic_0_ip_test basic_0_ip_test.cpp)
add_poponnx_cpp_unit_test(series_0_ip_test series_0_ip_test.cpp)
add_poponnx_cpp_unit_test(priorities_0_ip_test priorities_0_ip_test.cpp)
add_poponnx_cpp_unit_test(concat_0_ip_test concat_0_ip_test.cpp)
