# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
'''
Demonstrate how to call the same subgraph from a Python function with a repeat operation.
'''
import numpy as np
import popxl
import popxl.ops as ops

# Creating a model with popxl
ir = popxl.Ir()
main = ir.main_graph


# Op begin
def increment_fn(x: popxl.Tensor, value: popxl.Tensor):
    return x + value


with main:
    # host load
    x = popxl.variable(np.ones([2, 2], np.float32), name="x")
    value = popxl.variable(np.ones(x.shape, x.dtype.as_numpy()), name="value")

    # create graph
    increment_graph = ir.create_graph(increment_fn, x, value)

    # call graph in a loop
    o, = ops.repeat(increment_graph, 2, x, value)
    # Op end
    # host store
    o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="output_stream")
    ops.host_store(o_d2h, o)

with popxl.Session(ir, "ipu_model") as session:
    outputs = session.run()

np.testing.assert_allclose(outputs[o_d2h], [[3, 3], [3, 3]])
