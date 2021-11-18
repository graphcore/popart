# Tests of streaming memory ops and execution phase logic
# to run only these tests with ctest, consider using something like,
# `ctest -R streamingmemory -j5 -VV'.

add_popart_cpp_unit_test(overlap_compute_exchange_test overlap_compute_exchange_test.cpp VARIANTS "Hw")
add_popart_cpp_unit_test(executionphase_gradsumdec_test executionphase_gradsumdec_test.cpp)
add_popart_cpp_unit_test(remotebuffer_test remotebuffer_test.cpp VARIANTS "Hw")
add_popart_cpp_unit_test(executionphase_sharding_test executionphase_sharding_test.cpp VARIANTS "Cpu")
add_popart_cpp_unit_test(executionphase_initop_accumulator_test executionphase_initop_accumulator_test.cpp VARIANTS "IpuModel")
add_popart_cpp_unit_test(streamingmemoryopinserter_test streamingmemoryopinserter_test.cpp VARIANTS "Cpu")
add_popart_cpp_unit_test(host_stream_op_schedule_test host_stream_op_schedule_test.cpp VARIANTS "IpuModel")

add_popart_py_unit_test(streamingmemory_test_0 streamingmemory_test_0.py MATCHEXPR "test_weight_update or test_onchip_memory" VARIANTS "Hw")
add_popart_py_unit_test(streamingmemory_test_0 streamingmemory_test_0.py MATCHEXPR "test_replicated_sgd0_weight_update" VARIANTS "Hw")
add_popart_py_unit_test(streamingmemory_test_0 streamingmemory_test_0.py MATCHEXPR "test_replicated_sgd1and2_weight_update" VARIANTS "Hw")
add_popart_py_unit_test(streamingmemory_test_0 streamingmemory_test_0.py MATCHEXPR "test_replicated_adam_weight_update" VARIANTS "Hw")
add_popart_py_unit_test(streamingmemory_test_0 streamingmemory_test_0.py MATCHEXPR "test_replicated_lamb_weight_update" VARIANTS "Hw")

add_popart_py_unit_test(streamingmemory_test_1 streamingmemory_test_1.py VARIANTS "Hw")
add_popart_py_unit_test(streamingmemory_test_2 streamingmemory_test_2.py VARIANTS "Hw")
add_popart_py_unit_test(streamingmemory_tensor_location_test streamingmemory_tensor_location_test.py VARIANTS "Hw")
add_popart_py_unit_test(streamingmemory_attention_test streamingmemory_attention_test.py VARIANTS "Hw")

# T26754 && T26968
#add_popart_py_unit_test(streamingmemory_overlap_test streamingmemory_overlap_test.py VARIANTS "Hw")
add_popart_py_unit_test(host_stream_test host_stream_test.py VARIANTS "IpuModel")