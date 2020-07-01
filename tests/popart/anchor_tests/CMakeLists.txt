# Tests of pipelining
# to run only these tests with ctest, consider using something like,
# `ctest -R anchors -j5 -VV'.

add_popart_cpp_unit_test(
        weights_as_anchors_test_0 weights_as_anchors_test_0.cpp VARIANTS "IpuModel")
add_popart_py_unit_test(anchor_output_test VARIANTS Hw)
add_popart_py_unit_test(anchor_shape_tests VARIANTS Hw)