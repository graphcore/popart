# Tests of session API robustness to user errors
# to run only these tests with ctest, consider using something like,
# `ctest -R session_api -j5 -VV'.

add_popart_cpp_unit_test(basic_0_session_api_test basic_0_session_api_test.cpp)

add_popart_py_unit_test(reset_host_weights_test)
add_popart_py_unit_test(rng_test VARIANTS Hw)
