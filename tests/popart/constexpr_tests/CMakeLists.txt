# Tests of const expression folding, testing both
# (1) that the constant expressions (ce's) are correctly found, and
# (2) that the outputs are correctly computed statically on host.
# to run only these tests with ctest, consider using something like,
# `ctest -R ce_test -j5 -VV'. Hope that works. 

add_popart_cpp_unit_test(cast_ce_test cast_ce_test.cpp)
add_popart_cpp_unit_test(scale_ce_test scale_ce_test.cpp)
add_popart_cpp_unit_test(elementwise_ce_test elementwise_ce_test.cpp)
add_popart_cpp_unit_test(transpose_ce_test transpose_ce_test.cpp)
add_popart_cpp_unit_test(slice_ce_test slice_ce_test.cpp)
add_popart_cpp_unit_test(concat_ce_test concat_ce_test.cpp)
add_popart_cpp_unit_test(unsqueeze_ce_test unsqueeze_ce_test.cpp)
add_popart_cpp_unit_test(gemm_decomposition_ce_test gemm_decomposition_ce_test.cpp)
add_popart_cpp_unit_test(gather_ce_test gather_ce_test.cpp)
add_popart_cpp_unit_test(floor_ce_test floor_ce_test.cpp)

add_popart_py_unit_test(no_impl_test)
add_popart_py_unit_test(test_cast_ce)
add_popart_py_unit_test(test_squeeze_ce)
