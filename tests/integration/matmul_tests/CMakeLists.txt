# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Tests of ONNX matmul

add_unit_test(basic0_matmul_test
                         basic0_matmul_test.cpp)

add_unit_test(dataless0_train_matmul_test
                          dataless0_train_matmul_test.cpp)

add_unit_test(matmulgradpattern_test
                          matmulgradpattern_test.cpp)

add_popart_py_unit_test(matmul_partials_tests VARIANTS IpuModel2)
