# Tests of streaming memory ops and execution phase logic
# to run only these tests with ctest, consider using something like,
# `ctest -R streamingmemory -j5 -VV'.

add_popart_cpp_unit_test(overlap_compute_exchange_test overlap_compute_exchange_test.cpp VARIANTS "Hw")
add_popart_cpp_unit_test(executionphase_gradsumdec_test executionphase_gradsumdec_test.cpp)
add_popart_cpp_unit_test(remotebuffer_test remotebuffer_test.cpp VARIANTS "Hw")
add_popart_cpp_unit_test(executionphase_sharding_test executionphase_sharding_test.cpp VARIANTS "Cpu")
add_popart_cpp_unit_test(executionphase_initop_accumulator_test executionphase_initop_accumulator_test.cpp VARIANTS "IpuModel")

add_popart_py_unit_test(streamingmemory_test VARIANTS "Hw")
add_popart_py_unit_test(streamingmemory_tensor_location_test VARIANTS "Hw")
add_popart_py_unit_test(streamingmemory_attention_test VARIANTS "Hw")
