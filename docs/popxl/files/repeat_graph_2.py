# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
'''
Demonstrate how num_host_transfers works with host_load and host_store ops inside a repeat op.
'''

import numpy as np
import popxl
import popxl.ops as ops
from typing import Tuple

# SessionRun3 begin
# Creating a model with popxl
ir = popxl.Ir()
main = ir.main_graph

_INPUT_SHAPE = [2, 2]
_REPEAT_COUNT = 8


class Linear(popxl.Module):
    def __init__(self, x_h2d_: popxl.HostToDeviceStream,
                 y_d2h_: popxl.DeviceToHostStream):
        self.x_h2d = x_h2d_
        self.y_d2h = y_d2h_
        self.W: popxl.Tensor = None
        self.b: popxl.Tensor = None

    def build(self, x: popxl.Tensor, out_features: int,
              bias: bool = True) -> Tuple[popxl.Tensor, ...]:

        x = ops.host_load(self.x_h2d, "x")
        self.W = popxl.graph_input((x.shape[-1], out_features), popxl.float32,
                                   "W")
        y = x @ self.W
        if bias:
            self.b = popxl.graph_input((out_features, ), popxl.float32, "b")
            y = y + self.b

        ops.host_store(self.y_d2h, y)
        return y


with main:
    # host load
    x_h2d = popxl.h2d_stream(_INPUT_SHAPE, popxl.float32, name="x_stream")
    y_d2h = popxl.d2h_stream(_INPUT_SHAPE, popxl.float32, name="y_stream")
    W_data = np.ones([2, 2], np.float32)
    b_data = np.ones([2], np.float32)
    W = popxl.variable(W_data, name="W")
    b = popxl.variable(b_data, name="b")

    # This is the loop carried input.
    x = ops.init([2, 2], popxl.float32, "init")

    # Create graph, pass in streams
    linear = Linear(x_h2d, y_d2h)
    linear_graph = ir.create_graph(linear, x, out_features=2)

    # call graph in a loop
    # the x, W, b will be copied to the input of the `linear_graph` before the first iteration
    # the outputs of each iteration will be copied to the inputs of the next iteration
    # The outputs of the last iteration serve as the output of the `repeat` op
    # Note the iterations of 8, which we will also use as the num_host_transfers
    o, = ops.repeat(linear_graph,
                    _REPEAT_COUNT,
                    x,
                    inputs_dict={
                        linear.W: W,
                        linear.b: b
                    })

# The ost_load and host_store ops are both run _REPEAT_COUNT number of times, so set num_host_transfers
# to _REPEAT_COUNT.
ir.num_host_transfers = _REPEAT_COUNT

# Note the input shape here (_REPEAT_COUNT, *data_shape):
x_data = np.random.random([_REPEAT_COUNT, 2, 2]).astype(np.float32)
input_ = {x_h2d: x_data}
session = popxl.Session(ir, "ipu_model")

outputs = session.run(input_)
# SessionRun3 end

y_data: np.ndarray = outputs[y_d2h]

for i in range(_REPEAT_COUNT):
    x_data_i = x_data[i, ...]
    y_data_i = y_data[i, ...]
    np.testing.assert_allclose((x_data_i @ W_data) + b_data, y_data_i)
