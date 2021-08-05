# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# TODO(T42953): Make all tests run on IpuModel once T42732 solved.
add_popart_py_unit_test(tiedgathertest testtiedgather.py VARIANTS IpuModel MATCHEXPR "not test_tied_gather_pattern_correctness")
add_popart_py_unit_test(tiedgathertest__pattern_correctness testtiedgather.py VARIANTS Hw MATCHEXPR "test_tied_gather_pattern_correctness")
