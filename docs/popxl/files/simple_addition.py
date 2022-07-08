# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""
The intention of this example is to show a simple example of addition using popxl.
"""

import numpy as np
import popxl
import popxl.ops as ops

# Creating a model with popxl
ir = popxl.Ir()
main = ir.main_graph
with main:
    # host load
    input0 = popxl.h2d_stream([1], popxl.float32, name="input0_stream")
    a = ops.host_load(input0, "a")
    input1 = popxl.h2d_stream([1], popxl.float32, name="input1_stream")
    b = ops.host_load(input1, "b")

    # addition
    o = ops.add(a, b)

    # host store
    o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="output_stream")
    ops.host_store(o_d2h, o)

session = popxl.Session(ir, "ipu_model")

# Generate some random input data
data_a = np.random.rand(1).astype(np.float32)
data_b = np.random.rand(1).astype(np.float32)
inputs = {input0: data_a, input1: data_b}

# SessionRun begin
# run the model
with session:
    outputs = session.run(inputs)

print(f"Input a is {data_a}")
print(f"Input b is {data_b}")
print(f"Result is {outputs[o_d2h]}")
# SessionRun end

# SessionRun2 begin
# Alternatively:
run_output = {o_d2h: np.empty(shape=[1]).astype(np.float32)}

with session:
    session.run_with_outputs(inputs, run_output)

print(f"Input a is {data_a}")
print(f"Input b is {data_b}")
print(f"Result is {run_output[o_d2h]}")
# SessionRun2 end
