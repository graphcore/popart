# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Tests of inplace / view-changing logic
# to run only these tests with ctest, consider using something like,
# `ctest -R ip_test -j5 -VV'.

add_unit_test(schedule_liveness_0_test.cpp schedule_liveness_0_test.cpp)

add_popart_py_unit_test(serialization_to_poprithms_shift_graph_test0)
