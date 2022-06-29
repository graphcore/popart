# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
add_popart_cpp_unit_test(dropout_on_weight_test_0
                         dropout_on_weight_test_0.cpp VARIANTS "IpuModel;Hw")

add_popart_py_unit_test(dropout_outlinable_test)
