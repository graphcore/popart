# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
add_unit_test(dropout_on_weight_test_0
                         dropout_on_weight_test_0.cpp VARIANTS "IpuModel2;Hw")

add_popart_py_unit_test(dropout_outlinable_test)
