# Tests of pipelining
# to run only these tests with ctest, consider using something like,
# `ctest -R pipeline -j5 -VV'.

add_popart_cpp_unit_test(pipeline_topo_con_test_0 pipeline_topo_con_test_0.cpp VARIANTS "IpuModel")
add_popart_cpp_unit_test(pipeline_continuous_equivalent_test_0 
                         pipeline_continuous_equivalent_test_0.cpp VARIANTS "IpuModel")
add_popart_cpp_unit_test(pipeline_discontiguous_ipu_copy_test_0
                         pipeline_discontiguous_ipu_copy_test_0.cpp  VARIANTS "IpuModel")
add_popart_cpp_unit_test(pipeline_discontiguous_ipu_copy_test_1
                         pipeline_discontiguous_ipu_copy_test_1.cpp VARIANTS "IpuModel")
add_popart_cpp_unit_test(pipeline_quadratic_epsilon_test_0
                         pipeline_quadratic_epsilon_test_0.cpp VARIANTS "IpuModel")
add_popart_cpp_unit_test(pipeline_quadratic_epsilon_test_1
                         pipeline_quadratic_epsilon_test_1.cpp VARIANTS "IpuModel")
add_popart_cpp_unit_test(pipeline_recompute_numerical_test_0
                         pipeline_recompute_numerical_test_0.cpp VARIANTS "IpuModel")
add_popart_cpp_unit_test(pipeline_anchor_recomputed_tensor_check
                         pipeline_anchor_recomputed_tensor_check.cpp VARIANTS "IpuModel")
add_popart_cpp_unit_test(pipeline_no_multi_source_copies_test_0
                         pipeline_no_multi_source_copies_test_0.cpp VARIANTS "IpuModel")
add_popart_cpp_unit_test(pipeline_recompute_ir_test_0
                         pipeline_recompute_ir_test_0.cpp VARIANTS "IpuModel")
add_popart_cpp_unit_test(pipeline_recompute_ir_test_1
                         pipeline_recompute_ir_test_1.cpp VARIANTS "IpuModel")
add_popart_cpp_unit_test(pipeline_recompute_ir_test_2
                         pipeline_recompute_ir_test_2.cpp VARIANTS "IpuModel")

add_all_py_unit_tests("")
