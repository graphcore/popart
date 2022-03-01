# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

add_popart_py_unit_test(popxl-pipelining-serial_inference test_0_serial_inference.py VARIANTS IpuModel)
add_popart_py_unit_test(popxl-pipelining-pipelined_inference test_1_pipelined_inference.py VARIANTS IpuModel)
