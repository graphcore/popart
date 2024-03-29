# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
add_unit_test(sgd_mixed_mode_test_cpp_0 sgd_mixed_mode_test_cpp_0.cpp VARIANTS IpuModel2)
add_unit_test(sgd_mixed_mode_test_cpp_1_10 sgd_mixed_mode_test_cpp_1_10.cpp VARIANTS IpuModel2)
add_unit_test(sgd_mixed_mode_test_cpp_1_9 sgd_mixed_mode_test_cpp_1_9.cpp VARIANTS IpuModel2)
add_unit_test(sgd_mixed_mode_test_cpp_1_8 sgd_mixed_mode_test_cpp_1_8.cpp VARIANTS IpuModel2)
add_unit_test(sgd_mixed_mode_test_cpp_1_7 sgd_mixed_mode_test_cpp_1_7.cpp VARIANTS IpuModel2)
add_unit_test(sgd_mixed_mode_test_cpp_1_6 sgd_mixed_mode_test_cpp_1_6.cpp VARIANTS IpuModel2)
add_unit_test(sgd_mixed_mode_test_cpp_1_5 sgd_mixed_mode_test_cpp_1_5.cpp VARIANTS IpuModel2)
add_unit_test(sgd_mixed_mode_test_cpp_1_4 sgd_mixed_mode_test_cpp_1_4.cpp VARIANTS IpuModel2)
add_unit_test(sgd_mixed_mode_test_cpp_1_3 sgd_mixed_mode_test_cpp_1_3.cpp VARIANTS IpuModel2)
add_unit_test(sgd_mixed_mode_test_cpp_1_2 sgd_mixed_mode_test_cpp_1_2.cpp VARIANTS IpuModel2)
add_unit_test(sgd_mixed_mode_test_cpp_1_1 sgd_mixed_mode_test_cpp_1_1.cpp VARIANTS IpuModel2)
add_unit_test(sgd_mixed_mode_test_cpp_1_0 sgd_mixed_mode_test_cpp_1_0.cpp VARIANTS IpuModel2)
add_unit_test(sgd_mixed_mode_compatibility_test_0 sgd_mixed_mode_compatibility_test_0.cpp VARIANTS IpuModel2)

add_popart_py_unit_test(adam_mixed_mode_test_py_0)
add_popart_py_unit_test(adaptive_mixed_mode_test_py_0)
add_popart_py_unit_test(pytorch_comparisons VARIANTS IpuModel2)
add_popart_py_unit_test(rmsprop_tf_mode_test NUM_WORKERS auto)
add_popart_py_unit_test(sgd_mixed_mode_test_py_0)
add_popart_py_unit_test(sgd1_accumulator_test)

# sgd_mixed_mode_test_py_1 starts here
set(accl_types "accl1Type0" "accl1Type1")
set(accum_types "accumType0" "accumType1")
set(sgdAccMm_types "sgdAccMm0" "sgdAccMm1")
foreach(accl1Type ${accl_types})
  foreach(accumType ${accum_types})
    foreach(sgdAccMm ${sgdAccMm_types})
# linter complains if this is indented
add_popart_py_unit_test(sgd_mixed_mode_test_py_1 MATCHEXPR "[${accl1Type}-${accumType}-${sgdAccMm}]")
    endforeach()
  endforeach()
endforeach()
add_popart_py_unit_test(sgd_mixed_mode_test_py_1 MATCHEXPR "test_sgd_mixed_mode_1")
# sgd_mixed_mode_test_py_1 ends here

# global_batch_size_test starts here
set(batchserial_options "Unroll" "Loop")
set(explicit_loops_options "True" "False")
set(reduction_types "Sum" "MeanRunning" "MeanPost")
set(optimizers "SGD" "SGDM1" "SGDM2" "ADAM")

foreach(optimizer ${optimizers})
  foreach(reduction_type ${reduction_types})
    foreach(batchserial ${batchserial_options})
      foreach(explicit_loops ${explicit_loops_options})
# linter complains if this is indented
add_popart_py_unit_test(global_batch_size_test VARIANTS "${HW_EXCLUDE_C600}" MATCHEXPR "[${optimizer}-${reduction_type}-${batchserial}-${explicit_loops}]")
      endforeach()
    endforeach()
  endforeach()
endforeach()
# global_batch_size_test ends here
