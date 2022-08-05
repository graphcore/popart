# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Tests of transformation logic
# to run only these tests with ctest, consider using something like,
# `ctest -R transformation -j5 -VV'.

add_unit_test(remove_unused_input_transformation_test_0
       remove_unused_input_transformation_test_0.cpp)

add_unit_test(prepare_nodes_for_training_transformation_test_0
       prepare_nodes_for_training_transformation_test_0.cpp)

add_unit_test(pipelining_transform pipelining_transform.cpp VARIANTS "IpuModel2")

add_unit_test(explicit_recomputation explicit_recomputation_test.cpp)

add_unit_test(batchserialize_ir batchserialize_ir_test.cpp)

add_unit_test(mergeexchange mergeexchange_test.cpp VARIANTS "Hw")

add_unit_test(running_mean_counter running_mean_counter_test.cpp VARIANTS "IpuModel2")

add_unit_test(automatic_loss_scaling automatic_loss_scaling_test.cpp)

add_subdirectory(mergevarupdates)

add_popart_py_unit_test(serializematmul)
add_popart_py_unit_test(explicitrecomputation VARIANTS "IpuModel2")
add_popart_py_unit_test(batchserialize)
add_popart_py_unit_test(batchserialize_dropout_test VARIANTS Hw)

# We run this test on the POD builder only to load balance across builders, as
# it is a long test.
add_popart_py_unit_test(overlapio VARIANTS Hw LABELS multicard)
