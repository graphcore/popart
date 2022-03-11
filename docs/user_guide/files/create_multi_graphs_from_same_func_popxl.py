# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
'''
The intention of this example is to show how to create multiple
subgraphs from the same function.
'''

import numpy as np
import popxl
import popxl.ops as ops

# Creating a model with popxl
ir = popxl.Ir()
main = ir.main_graph


# Op begin
def matmul_fn(x: popxl.Tensor, w: popxl.Tensor):
    return x @ w


with main:
    # host load
    input = popxl.h2d_stream([2, 2], popxl.float32, name="input_stream")
    x = ops.host_load(input, "x")

    w1 = popxl.variable(np.ones(x.shape, x.dtype.as_numpy()), name="w1")
    w2 = popxl.variable(np.ones(x.shape[-1], x.dtype.as_numpy()), name="w2")

    # create two graphs
    matmul_graph1 = ir.create_graph(matmul_fn, x, w1)
    matmul_graph2 = ir.create_graph(matmul_fn, x, w2)

    # Op end

    # call graph
    o, = ops.call(matmul_graph1, x, w1)
    o, = ops.call(matmul_graph2, o, w2)

    # host store
    o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="output_stream")
    ops.host_store(o_d2h, o)

session = popxl.Session(ir, "ipu_model")

# Generate some random input data
inputs = {input: np.random.rand(2, 2).astype(np.float32)}

# run the model
outputs = session.run(inputs)

print(f"Input is {inputs[input]}")
print(f"Result is {outputs[o_d2h]}")
