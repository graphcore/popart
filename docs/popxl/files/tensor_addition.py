# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
'''
The intention of this example is to show the three different types of tensors in popxl.
'''

import popxl
import popxl.ops as ops

# Creating a model with popxl
ir = popxl.Ir()
main = ir.main_graph
# Op begin
with main:
    a = popxl.variable(3, dtype=popxl.int8, name="variable_a")
    b = popxl.constant(1, dtype=popxl.int8, name="constant_b")

    # addition
    o = a + b
    # Op end
    # host store
    o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="output_stream")
    ops.host_store(o_d2h, o)

# Session begin
# Construct an Ir `ir`...

ir.num_host_transfers = 1

with popxl.Session(ir, "ipu_model") as session:
    outputs = session.run()
# Session end

print(f"Result is {outputs[o_d2h]}")
