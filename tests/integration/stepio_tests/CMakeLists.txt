# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
add_unit_test(stepio_cpp_tests_0 stepio_cpp_tests_0.cpp)
add_unit_test(stepio_nelms_error_test stepio_nelms_error_test.cpp)
add_unit_test(stepiosplitter_test stepiosplitter_test.cpp)

add_popart_py_unit_test(stepio_tests_py VARIANTS Hw)
add_popart_py_unit_test(stepio_tests_py_cpu)
