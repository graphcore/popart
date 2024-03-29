# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Tests of streaming memory ops and execution phase logic
# to run only these tests with ctest, consider using something like,
# `ctest -R streamingmemory -j5 -VV'.

add_unit_test(overlap_compute_exchange_test overlap_compute_exchange_test.cpp VARIANTS "Hw")
add_unit_test(executionphase_gradsumdec_test executionphase_gradsumdec_test.cpp)
add_unit_test(remotebuffer_test remotebuffer_test.cpp VARIANTS "${HW_EXCLUDE_C600}")
add_unit_test(executionphase_sharding_test executionphase_sharding_test.cpp VARIANTS "Cpu")
add_unit_test(executionphase_initop_accumulator_test executionphase_initop_accumulator_test.cpp VARIANTS "IpuModel2")
add_unit_test(streamingmemoryopinserter_test streamingmemoryopinserter_test.cpp VARIANTS "Cpu")
add_unit_test(host_stream_op_schedule_test host_stream_op_schedule_test.cpp VARIANTS "IpuModel2")
add_unit_test(functionbuffers_test functionbuffers_test.cpp VARIANTS "IpuModel2")
add_unit_test(functionbuffers_test_nested functionbuffers_test_nested.cpp VARIANTS "IpuModel2")

add_popart_py_unit_test(streamingmemory_test_0 MATCHEXPR "test_weight_update or test_onchip_memory" VARIANTS "Hw")
add_popart_py_unit_test(streamingmemory_test_0 MATCHEXPR "test_replicated_sgd0_weight_update" VARIANTS "${HW_EXCLUDE_C600}")
add_popart_py_unit_test(streamingmemory_test_0 MATCHEXPR "test_replicated_sgd1and2_weight_update" VARIANTS "${HW_EXCLUDE_C600}")
add_popart_py_unit_test(streamingmemory_test_0 MATCHEXPR "test_replicated_adam_weight_update" VARIANTS "${HW_EXCLUDE_C600}")
add_popart_py_unit_test(streamingmemory_test_0 MATCHEXPR "test_replicated_lamb_weight_update" VARIANTS "${HW_EXCLUDE_C600}")
add_popart_py_unit_test(streamingmemory_test_0 MATCHEXPR "test_gradient_clipping" VARIANTS "${HW_EXCLUDE_C600}")

add_popart_py_unit_test(streamingmemory_test_1 VARIANTS "${HW_EXCLUDE_C600}")
add_popart_py_unit_test(streamingmemory_test_2 VARIANTS "${HW_EXCLUDE_C600}")
add_popart_py_unit_test(streamingmemory_tensor_location_test VARIANTS "Hw")
add_popart_py_unit_test(streamingmemory_attention_test VARIANTS "${HW_EXCLUDE_C600}")

# TODO T26754 && T26968
add_popart_py_unit_test(streamingmemory_overlap_test VARIANTS "Hw" PROPERTIES DISABLED TRUE)
add_popart_py_unit_test(host_stream_test VARIANTS "IpuModel2")
