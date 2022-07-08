# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""
The intention of this example is to show how to write and read tensor data.
"""

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

some_new_np_data = np.array(5).astype(np.int32)

# TensorData begin
session = popxl.Session(ir, "ipu_model")

# Allowed when detached, device weights will be updated on next context enter
session.write_variable_data(a, some_new_np_data)

with session:
    # No weights_to_host as no `run` has invalidated the host weights yet
    a_data = session.get_tensor_data(a)

    outputs = session.run()

    # No weights_to_host, is constant
    b_data = session.get_tensor_data(b)
    # Causes weights_to_host as host weights currently out-of-sync
    datas = session.get_tensors_data([a, b])

    # Performs weights_from_host
    session.write_variable_data(a, some_new_np_data)

    # No weights_to_host, write_variable_data did not invalidate
    a_data = session.get_tensor_data(a)

    outputs = session.run()

# Leave context, causes detach and weights_to_host

# No weights_to_host as detached. Latest weights were already fetched due to
# leaving context
tensor_datas = session.get_tensors_data([a, b])

# TensorData end
