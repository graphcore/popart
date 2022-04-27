# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
'''
Demonstrate how to call the same subgraph with repeat op and `inputs_dict`.

The argument `inputs_dict` connects the inputs of the subgraph with the caller graph.
'''

import numpy as np
import popxl
import popxl.ops as ops
from typing import Tuple

# Creating a model with popxl
ir = popxl.Ir()
main = ir.main_graph


# Op begin
class Linear(popxl.Module):
    def __init__(self):
        self.W: popxl.Tensor = None
        self.b: popxl.Tensor = None

    def build(self, x: popxl.Tensor, out_features: int,
              bias: bool = True) -> Tuple[popxl.Tensor, ...]:
        self.W = popxl.graph_input((x.shape[-1], out_features), popxl.float32,
                                   "W")
        y = x @ self.W
        if bias:
            self.b = popxl.graph_input((out_features, ), popxl.float32, "b")
            y = y + self.b
        return y


with main:
    # host load
    x = popxl.variable(np.ones([2, 2], np.float32), name="x")
    W = popxl.variable(np.ones([2, 2], np.float32), name="W")
    b = popxl.variable(np.ones([2], np.float32), name="b")

    # create graph
    linear = Linear()
    linear_graph = ir.create_graph(linear, x, out_features=2)

    # call graph in a loop
    # the x, W, b will be copied to the input of the `linear_graph` before the first iteration
    # the outputs of each iteration will be copied to the inputs of the next iteration
    # The outputs of the last iteration serve as the output of the `repeat` op
    o, = ops.repeat(linear_graph, 2, x, inputs_dict={linear.W: W, linear.b: b})
    # Op end
    # host store
    o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="output_stream")
    ops.host_store(o_d2h, o)

with popxl.Session(ir, "ipu_model") as session:
    outputs = session.run()

np.testing.assert_allclose(outputs[o_d2h], [[7, 7], [7, 7]])
