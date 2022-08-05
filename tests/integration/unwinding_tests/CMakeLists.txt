# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Tests of transformation logic
# to run only these tests with ctest, consider using something like,
# `ctest -R transformation -j5 -VV'.

add_unit_test(unwinding_test0 unwinding_test0.cpp)
