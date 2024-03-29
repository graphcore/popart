# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""
The intention of this example is to show how to get gradients with autodiff.
"""

import numpy as np
import popxl
import popxl.ops as ops
from typing import Tuple
import popxl.transforms as transforms

# Creating a model with popxl
ir = popxl.Ir()
main = ir.main_graph


# Op begin
class Linear(popxl.Module):
    def __init__(self):
        self.W: popxl.Tensor = None
        self.b: popxl.Tensor = None

    def build(
        self, x: popxl.Tensor, out_features: int, bias: bool = True
    ) -> Tuple[popxl.Tensor, ...]:
        self.W = popxl.graph_input((x.shape[-1], out_features), popxl.float32, "W")
        y = x @ self.W
        if bias:
            self.b = popxl.graph_input((out_features,), popxl.float32, "b")
            y = y + self.b
        return y


with main:
    # host load
    input = popxl.h2d_stream([2, 2], popxl.float32, name="input_stream")
    x = ops.host_load(input, "x")
    W_data = np.random.normal(0, 0.1, (2, 2)).astype(np.float32)
    W = popxl.variable(W_data, name="W")
    b_data = np.random.normal(0, 0.4, (2)).astype(np.float32)
    b = popxl.variable(b_data, name="b")

    # create graph
    linear = Linear()
    linear_graph = ir.create_graph(linear, x, out_features=2)

    fwd_call_info = ops.call_with_info(
        linear_graph, x, inputs_dict={linear.W: W, linear.b: b}
    )
    y = fwd_call_info.outputs[0]

    # get the gradients from autodiff
    bwd_graph_info = transforms.autodiff(linear_graph)
    grad_seed = popxl.constant(np.ones((2, 2), np.float32))
    activations = bwd_graph_info.inputs_dict(fwd_call_info)
    grads_x, grads_w, grads_b = ops.call(
        bwd_graph_info.graph, grad_seed, inputs_dict=activations
    )

    # host store
    o_d2h = popxl.d2h_stream(y.shape, y.dtype, name="output_stream")
    ops.host_store(o_d2h, y)

    grad_d2h = popxl.d2h_stream(grads_w.shape, grads_w.dtype, name="grad_stream")
    ops.host_store(grad_d2h, grads_w)
    # Op end

with popxl.Session(ir, "ipu_model") as session:
    # Generate some random input data
    inputs = {input: np.random.rand(2, 2).astype(np.float32)}
    outputs = session.run(inputs)

print(f"Input is {inputs[input]}")
print(f"Output is {outputs[o_d2h]}")
print(f"Grads is {outputs[grad_d2h]}")
