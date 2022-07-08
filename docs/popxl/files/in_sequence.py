# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""
The intention of this example is to show how to use PopXL context manager in_sequence.
"""

import popxl
import popxl.ops as ops

# Creating a model with popxl
ir = popxl.Ir()
main = ir.main_graph
# Op begin
with main:
    x = popxl.variable(1, popxl.float32)
    b = popxl.constant(5, popxl.float32)

    with popxl.in_sequence():
        ops.var_updates.copy_var_update_(x, b)
        # host store
        o_d2h = popxl.d2h_stream(x.shape, x.dtype, name="output_stream")
        ops.host_store(o_d2h, x)

# run the model
with popxl.Session(ir, "ipu_model") as session:
    outputs = session.run()
