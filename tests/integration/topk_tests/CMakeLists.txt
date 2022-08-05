# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Tests of ONNX TopK Operator
# to run only these tests with ctest, consider using something like,
# `ctest -R topk -j5 -VV'.

add_unit_test(basic0_topk_test basic0_topk_test.cpp)
add_unit_test(train0_topk_test train0_topk_test.cpp)
