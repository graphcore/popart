# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
add_unit_test(unittest_willow_op_call test_call.cpp)
add_unit_test(unittest_willow_op_dropout test_dropout.cpp)
add_unit_test(unittest_willow_op_reducegrad test_reducegrad.cpp)
add_unit_test(unittest_willow_op_addgrad test_addgrad.cpp)
add_unit_test(unittest_willow_op_remoteload test_remoteload.cpp)
add_unit_test(unittest_willow_op_batchnormattributes test_batchnorm_attributes.cpp)
