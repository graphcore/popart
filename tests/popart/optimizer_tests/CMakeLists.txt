add_popart_cpp_unit_test(sgd_mixed_mode_test_cpp_0 sgd_mixed_mode_test_cpp_0.cpp VARIANTS IpuModel)
add_popart_cpp_unit_test(sgd_mixed_mode_test_cpp_1_10 sgd_mixed_mode_test_cpp_1_10.cpp)
add_popart_cpp_unit_test(sgd_mixed_mode_test_cpp_1_9 sgd_mixed_mode_test_cpp_1_9.cpp)
add_popart_cpp_unit_test(sgd_mixed_mode_test_cpp_1_8 sgd_mixed_mode_test_cpp_1_8.cpp)
add_popart_cpp_unit_test(sgd_mixed_mode_test_cpp_1_7 sgd_mixed_mode_test_cpp_1_7.cpp)
add_popart_cpp_unit_test(sgd_mixed_mode_test_cpp_1_6 sgd_mixed_mode_test_cpp_1_6.cpp)
add_popart_cpp_unit_test(sgd_mixed_mode_test_cpp_1_5 sgd_mixed_mode_test_cpp_1_5.cpp)
add_popart_cpp_unit_test(sgd_mixed_mode_test_cpp_1_4 sgd_mixed_mode_test_cpp_1_4.cpp)
add_popart_cpp_unit_test(sgd_mixed_mode_test_cpp_1_3 sgd_mixed_mode_test_cpp_1_3.cpp)
add_popart_cpp_unit_test(sgd_mixed_mode_test_cpp_1_2 sgd_mixed_mode_test_cpp_1_2.cpp)
add_popart_cpp_unit_test(sgd_mixed_mode_test_cpp_1_1 sgd_mixed_mode_test_cpp_1_1.cpp)
add_popart_cpp_unit_test(sgd_mixed_mode_test_cpp_1_0 sgd_mixed_mode_test_cpp_1_0.cpp)
add_popart_cpp_unit_test(sgd_mixed_mode_compatibility_test_0 sgd_mixed_mode_compatibility_test_0.cpp)

add_popart_py_unit_test(adam_mixed_mode_test_py_0)
add_popart_py_unit_test(adaptive_mixed_mode_test_py_0)
add_popart_py_unit_test(pytorch_comparisons VARIANTS IpuModel)
add_popart_py_unit_test(rmsprop_tf_mode_test)
add_popart_py_unit_test(sgd_mixed_mode_test_py_0)
add_popart_py_unit_test(sgd_mixed_mode_test_py_1)
add_popart_py_unit_test(sgd1_accumulator_test)

add_popart_py_unit_test(global_batch_size_test MATCHEXPR sgd_ VARIANTS Hw)
add_popart_py_unit_test(global_batch_size_test MATCHEXPR adam VARIANTS Hw)
add_popart_py_unit_test(global_batch_size_test MATCHEXPR lamb VARIANTS Hw)
# Further split the SGDM tests on the first parameter of the test function to,
# help reduce timeouts. This could break if the test functions change.
# NOTE: CMake bug: For some reason, the keywords get set wrong if VARIANTS comes
#       after MATCHEXPR
add_popart_py_unit_test(global_batch_size_test VARIANTS Hw MATCHEXPR "sgdm[True")
add_popart_py_unit_test(global_batch_size_test VARIANTS Hw MATCHEXPR "sgdm[False")
