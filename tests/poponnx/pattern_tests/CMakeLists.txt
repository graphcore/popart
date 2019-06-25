# Tests of pattern logic
# to run only these tests with ctest, consider using something like,
# `ctest -R patterns_test -j5 -VV'.

add_poponnx_cpp_unit_test(softmaxgraddirect_0_patterns_test 
                          softmaxgraddirect_0_patterns_test.cpp)

add_poponnx_cpp_unit_test(splitgather_0_patterns_test 
                          splitgather_0_patterns_test.cpp)

add_poponnx_cpp_unit_test(splitgather_default_patterns_test 
                          splitgather_default_patterns_test.cpp)

add_poponnx_cpp_unit_test(patterns_test 
                          patterns_test.cpp)

add_poponnx_cpp_unit_test(split_to_slice_patterns_test 
                          split_to_slice_patterns_test.cpp)


add_poponnx_cpp_unit_test(replicate_and_shard_ir_test_0 
                          replicate_and_shard_ir_test_0.cpp)

