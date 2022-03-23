# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
'''
The intention of this example is to show how to write and read tensor data.
'''

import popxl
import popxl.ops as ops
import numpy as np

# Creating a model with PopXL
ir = popxl.Ir()
main = ir.main_graph

with main:
    a = popxl.variable(3, dtype=popxl.int32, name="variable_a")
    b = popxl.constant(1, dtype=popxl.int32, name="constant_b")

    # addition
    o = a + b
    # host store
    o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="output_stream")
    ops.host_store(o_d2h, o)

# Construct an Ir `ir`...

ir.num_host_transfers = 1

# TensorData begin
session = popxl.Session(ir, "ipu_model")
outputs = session.run()

print(f"Result is {outputs[o_d2h]}")

a_data = session.get_tensor_data(a)
b_data = session.get_tensor_data(b)

# Check the device values of 'a', 'b'
assert a_data == np.array(3)
assert b_data == np.array(1)

# Write a new value for `a`
session.write_variable_data(a, np.array(5).astype(np.int32))

# Variable now updated on 'device'
assert session.get_tensor_data(a) == np.array(5)
# TensorData end
