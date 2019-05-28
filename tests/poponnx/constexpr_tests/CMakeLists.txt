# Tests of const expression folding, testing both
# (1) that the constant expressions (ce's) are correctly found, and
# (2) that the outputs are correctly computed statically on host.
# to run only these tests with ctest, consider using something like,
# `ctest -R ce_test -j5 -VV'. Hope that works. 

add_poponnx_cpp_unit_test(cast_ce_test cast_ce_test.cpp)
add_poponnx_cpp_unit_test(scale_ce_test scale_ce_test.cpp)
add_poponnx_cpp_unit_test(add_ce_test add_ce_test.cpp)
add_poponnx_cpp_unit_test(transpose_ce_test transpose_ce_test.cpp)
add_poponnx_cpp_unit_test(slice_ce_test slice_ce_test.cpp)
add_poponnx_cpp_unit_test(concat_ce_test concat_ce_test.cpp)
add_poponnx_cpp_unit_test(unsqueeze_ce_test unsqueeze_ce_test.cpp)
add_poponnx_cpp_unit_test(gemm_decomposition_ce_test gemm_decomposition_ce_test.cpp)
add_poponnx_cpp_unit_test(gather_ce_test gather_ce_test.cpp)

