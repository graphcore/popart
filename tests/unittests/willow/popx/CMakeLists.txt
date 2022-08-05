# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
add_unit_test(unittest_willow_popx_executablexserializer test_executablexserializer.cpp SUPPORT_LIBS popart_capnp)
add_unit_test(unittest_willow_popx_poptensors test_poptensors.cpp)
