# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
'''
The intention of this example is to show how to use
call_with_info op.
'''

import popxl
import popxl.ops as ops

# Creating a model with popxl
ir = popxl.Ir()
main = ir.main_graph


# Op begin
def increment_fn(x: popxl.Tensor):
    value = popxl.graph_input(x.shape, x.dtype, "value")
    # inplace increment of the input tensor
    ops.var_updates.copy_var_update_(x, x + value)


with main, popxl.in_sequence():
    x = popxl.variable(1)
    value1 = popxl.constant(1)

    # create graph
    increment_graph = ir.create_graph(increment_fn, x)
    # call graph
    info = ops.call_with_info(increment_graph, x, value1)
    info.set_parent_input_modified(x)
    # host store
    o_d2h = popxl.d2h_stream(x.shape, x.dtype, name="output_stream")
    ops.host_store(o_d2h, x)
    # Op end

session = popxl.Session(ir, "ipu_model")

outputs = session.run()

print(f"Result is {outputs[o_d2h]}")
