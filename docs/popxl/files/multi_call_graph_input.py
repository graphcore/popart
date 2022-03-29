# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
'''
Demonstrate how to use graph_input in a function and how to call the subgraph multiple times.
'''

import numpy as np
import popxl
import popxl.ops as ops

# Creating a model with popxl
ir = popxl.Ir()
main = ir.main_graph


# Op begin
def increment_fn(x: popxl.Tensor):
    value = popxl.graph_input(x.shape, x.dtype, "value")
    return x + value


with main:
    # host load
    input = popxl.h2d_stream([2, 2], popxl.float32, name="input_stream")
    x = ops.host_load(input, "x")

    # create graph
    increment_graph = ir.create_graph(increment_fn, x)

    # two variable values
    value1 = popxl.variable(np.ones(x.shape, x.dtype.as_numpy()),
                            name="value1")
    value2 = popxl.variable(2 * np.ones(x.shape, x.dtype.as_numpy()),
                            name="value2")

    # call graph
    o, = ops.call(increment_graph, x, value1)
    o, = ops.call(increment_graph, o, value2)
    # Op end

    # host store
    o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="output_stream")
    ops.host_store(o_d2h, o)

# Generate some random input data
inputs = {input: np.random.rand(2, 2).astype(np.float32)}
session = popxl.Session(ir, "ipu_model")

outputs = session.run(inputs)

print(f"Input is {inputs[input]}")
print(f"Result is {outputs[o_d2h]}")
