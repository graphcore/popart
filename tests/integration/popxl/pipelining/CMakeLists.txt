# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

add_popart_py_unit_test(test_0_serial_inference VARIANTS IpuModel)
add_popart_py_unit_test(test_1_pipelined_inference VARIANTS IpuModel)
