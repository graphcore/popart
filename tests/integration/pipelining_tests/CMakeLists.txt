# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

add_popart_py_unit_test(pipeline_full_recompute_test pipeline_full_recompute_test.py VARIANTS IpuModel)
add_popart_py_unit_test(pipeline_grad_accl_test pipeline_grad_accl_test.py VARIANTS IpuModel)
add_popart_py_unit_test(pipeline_multi_loss_numerical_0 pipeline_multi_loss_numerical_0.py VARIANTS IpuModel)
add_popart_py_unit_test(pipeline_test_train pipeline_test.py MATCHEXPR "test_output_matches_train" VARIANTS IpuModel)
add_popart_py_unit_test(pipeline_test_infer pipeline_test.py MATCHEXPR "test_output_matches_infer" VARIANTS IpuModel)
add_popart_py_unit_test(pipeline_test_rest pipeline_test.py MATCHEXPR "not test_output_matches_train and not test_output_matches_infer" VARIANTS IpuModel)
add_popart_py_unit_test(pipeline_dropout_test pipeline_dropout_test.py VARIANTS Hw)
add_popart_py_unit_test(replicated_pipeline_test replicated_pipeline_test.py VARIANTS Hw)
add_popart_py_unit_test(pipeline_boundary_test pipeline_boundary_test.py VARIANTS IpuModel)
add_popart_py_unit_test(pipeline_consecutive_stages_test pipeline_consecutive_stages_test.py VARIANTS IpuModel)
add_popart_py_unit_test(pipeline_multiple_restores_per_stash pipeline_multiple_restores_per_stash.py VARIANTS IpuModel)
add_popart_py_unit_test(pipeline_grad_tensor_consumed_by_multiple_pipeline_stages pipeline_grad_tensor_consumed_by_multiple_pipeline_stages.py VARIANTS IpuModel)
add_popart_py_unit_test(pipeline_user_recompute_test pipeline_user_recompute_test.py VARIANTS IpuModel)
add_popart_py_unit_test(explicit_pipeline_test explicit_pipeline_test.py VARIANTS IpuModel)
