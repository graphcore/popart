# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Tests of pattern logic
# to run only these tests with ctest, consider using something like,
# `ctest -R patterns_test -j5 -VV'.

add_unit_test(upsample_to_resize_pattern_test
                          upsample_to_resize_pattern_test.cpp)

add_unit_test(softmaxgraddirect_0_patterns_test
                          softmaxgraddirect_0_patterns_test.cpp)

add_unit_test(splitgather_0_patterns_test
                          splitgather_0_patterns_test.cpp VARIANTS "IpuModel2")

add_unit_test(splitgather_default_patterns_test
                          splitgather_default_patterns_test.cpp VARIANTS "IpuModel2")

add_unit_test(pattern_test
                          pattern_test.cpp)

add_unit_test(patterns_test
                          patterns_test.cpp)

add_unit_test(split_to_slice_patterns_test
                          split_to_slice_patterns_test.cpp VARIANTS "IpuModel2")

add_unit_test(replicate_and_shard_ir_test_0
                          replicate_and_shard_ir_test_0.cpp VARIANTS "IpuModel2")

add_unit_test(merge_duplicate_ops_test
                          merge_duplicate_ops_test.cpp)

add_unit_test(direct_view_change_test
                          direct_view_change_test.cpp)

add_popart_py_unit_test(test_excludes)

add_popart_py_unit_test(test_enable_patterns)

add_popart_py_unit_test(test_expandcast VARIANTS IpuModel2)

add_popart_py_unit_test(test_view_simplify_pattern VARIANTS IpuModel2)

add_popart_py_unit_test(test_foldmatmulpattern)

add_popart_py_unit_test(test_transpose_to_identity_or_reshape VARIANTS IpuModel2 MATCHEXPR test_replace_with_identity)
add_popart_py_unit_test(test_transpose_to_identity_or_reshape VARIANTS IpuModel2 MATCHEXPR test_replace_with_reshape)
add_popart_py_unit_test(test_transpose_to_identity_or_reshape VARIANTS IpuModel2 MATCHEXPR test_fail_due_to_non_trivial_reshape)
add_popart_py_unit_test(test_transpose_to_identity_or_reshape VARIANTS IpuModel2 MATCHEXPR test_fail_due_to_mismatch_permutation)
