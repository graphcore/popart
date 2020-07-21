# Tests of pingpong ops and logic
# to run only these tests with ctest, consider using something like,
# `ctest -R pingpong -j5 -VV'.

add_popart_cpp_unit_test(overlap_compute_exchange_test overlap_compute_exchange_test.cpp VARIANTS "Hw")
add_popart_cpp_unit_test(pingpong_gradsumdec_test pingpong_gradsumdec_test.cpp)
add_popart_cpp_unit_test(remotebuffer_test remotebuffer_test.cpp VARIANTS "Hw")
add_popart_cpp_unit_test(pingpong_sharding_test pingpong_sharding_test.cpp VARIANTS "Cpu")
add_popart_cpp_unit_test(pingpong_initop_accumulator_test pingpong_initop_accumulator_test.cpp VARIANTS "IpuModel")

add_popart_py_unit_test(pingpong_test VARIANTS "Hw")
add_popart_py_unit_test(pingpong_cache_test VARIANTS "Hw")
add_popart_py_unit_test(pingpong_attention_test VARIANTS "Hw")
