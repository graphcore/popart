# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Tests of ONNX If Operator
# to run only these tests with ctest, consider using something like,
# `ctest -R logical_if_test -j5 -VV'.

add_unit_test(builder_0_logical_if_test builder_0_logical_if_test.cpp)
add_unit_test(basic_0_logical_if_test basic_0_logical_if_test.cpp)
add_unit_test(namescope_0_logical_if_test namescope_0_logical_if_test.cpp)
add_unit_test(namescope_1_logical_if_test namescope_1_logical_if_test.cpp)
