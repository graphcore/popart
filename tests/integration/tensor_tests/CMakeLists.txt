# Tests of tensor layout and shape
# to run only these tests with ctest, consider using something like,
# `ctest -R ip_test -j5 -VV'.

add_popart_cpp_unit_test(region_0_test region_0_test.cpp)

