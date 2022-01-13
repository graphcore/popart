# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Testing the case where an Op does not have a path to it from a Stream Tensor
add_popart_cpp_unit_test(auto_virtual_graph_relu_on_weight_test_0
                          auto_virtual_graph_relu_on_weight_test_0.cpp VARIANTS "IpuModel")
