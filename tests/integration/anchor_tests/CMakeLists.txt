# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Tests of pipelining
# to run only these tests with ctest, consider using something like,
# `ctest -R anchors -j5 -VV'.

add_unit_test(
        weights_as_anchors_test_0 weights_as_anchors_test_0.cpp VARIANTS "IpuModel2")
add_popart_py_unit_test(anchor_output_test VARIANTS "${HW_EXCLUDE_C600}")
add_popart_py_unit_test(anchor_output_test_2 VARIANTS Hw)
add_popart_py_unit_test(anchor_shape_tests VARIANTS Hw)
