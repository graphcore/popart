# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
add_popart_cpp_unit_test(unittest_willow_op_call        test_call.cpp)
add_popart_cpp_unit_test(unittest_willow_op_dropout     test_dropout.cpp)
add_popart_cpp_unit_test(unittest_willow_op_castfromfp8 test_castfromfp8.cpp)
add_popart_cpp_unit_test(unittest_willow_op_casttofp8   test_casttofp8.cpp)
add_popart_cpp_unit_test(unittest_willow_op_reducegrad  test_reducegrad.cpp)
