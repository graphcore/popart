# Tests of transformation logic
# to run only these tests with ctest, consider using something like,
# `ctest -R transformation -j5 -VV'.

add_popart_cpp_unit_test(remove_unused_input_transformation_test_0
       remove_unused_input_transformation_test_0.cpp)

add_popart_cpp_unit_test(prepare_nodes_for_training_transformation_test_0
       prepare_nodes_for_training_transformation_test_0.cpp)

add_popart_cpp_unit_test(pipelining_transform pipelining_transform.cpp VARIANTS "IpuModel")

add_popart_cpp_unit_test(hostreduce_transform hostreduce_transformation_test.cpp VARIANTS "Hw")

add_popart_cpp_unit_test(explicit_recomputation explicit_recomputation_test.cpp)

add_popart_cpp_unit_test(batchserialize_ir batchserialize_ir_test.cpp)

add_popart_cpp_unit_test(mergeexchange mergeexchange_test.cpp VARIANTS "Hw")

add_subdirectory(mergevarupdates)

add_popart_py_unit_test(serializematmul serializematmul.py)
add_popart_py_unit_test(explicitrecomputation explicitrecomputation.py)
add_popart_py_unit_test(batchserialize batchserialize.py)
add_popart_py_unit_test(batchserialize_dropout_test batchserialize_dropout_test.py VARIANTS Hw)
