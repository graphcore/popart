# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
add_unit_test(popart._internal.ir_cpp--basic_graph basic_graph.cpp VARIANTS IpuModel2)
add_unit_test(popart._internal.ir_cpp--loop_accumulate loop_accumulate.cpp VARIANTS IpuModel2)
add_unit_test(popart._internal.ir_cpp--basic_inference basic_inference.cpp VARIANTS IpuModel2)
add_unit_test(popart._internal.ir_cpp--basic_training_loop basic_training_loop.cpp VARIANTS IpuModel2)
