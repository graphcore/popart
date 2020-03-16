# Tests of pingpong ops and logic
# to run only these tests with ctest, consider using something like,
# `ctest -R pingpong -j5 -VV'.

add_popart_cpp_unit_test(remotebuffer_test remotebuffer_test.cpp VARIANTS "Hw")
