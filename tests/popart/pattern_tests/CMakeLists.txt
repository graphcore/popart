# Tests of pattern logic
# to run only these tests with ctest, consider using something like,
# `ctest -R patterns_test -j5 -VV'.

add_popart_cpp_unit_test(upsample_to_resize_pattern_test 
                          upsample_to_resize_pattern_test.cpp)

add_popart_cpp_unit_test(softmaxgraddirect_0_patterns_test 
                          softmaxgraddirect_0_patterns_test.cpp)

add_popart_cpp_unit_test(splitgather_0_patterns_test 
                          splitgather_0_patterns_test.cpp VARIANTS "IpuModel")

add_popart_cpp_unit_test(splitgather_default_patterns_test 
                          splitgather_default_patterns_test.cpp VARIANTS "IpuModel")

add_popart_cpp_unit_test(pattern_test 
                          pattern_test.cpp)

add_popart_cpp_unit_test(patterns_test 
                          patterns_test.cpp)

add_popart_cpp_unit_test(split_to_slice_patterns_test 
                          split_to_slice_patterns_test.cpp VARIANTS "IpuModel")

add_popart_cpp_unit_test(replicate_and_shard_ir_test_0 
                          replicate_and_shard_ir_test_0.cpp VARIANTS "IpuModel")

add_popart_cpp_unit_test(merge_duplicate_ops_test 
                          merge_duplicate_ops_test.cpp)

add_popart_py_unit_test(test_excludes)

add_popart_py_unit_test(test_enable_patterns)

add_popart_py_unit_test(test_reshape_collapse_pattern VARIANTS IpuModel)

# TODO(T36508): Re-enable.
# add_popart_py_unit_test(test_transpose_to_identity_or_reshape VARIANTS IpuModel)
