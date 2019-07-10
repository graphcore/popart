# Tests of transformation logic
# to run only these tests with ctest, consider using something like,
# `ctest -R transformation -j5 -VV'.

add_poponnx_cpp_unit_test(remove_unused_input_transformation_test_0 
       remove_unused_input_transformation_test_0.cpp)

add_poponnx_cpp_unit_test(prepare_nodes_for_training_transformation_test_0 
       prepare_nodes_for_training_transformation_test_0.cpp)

add_poponnx_cpp_unit_test(pipelining_transform pipelining_transform.cpp)

add_subdirectory(mergevarupdates)

