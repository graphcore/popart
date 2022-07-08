# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""
This example shows how to use remote buffers and remote variables.
"""

import numpy as np

import popxl
import popxl.dtypes as dtypes
import popxl.ops as ops

# remote_var begin
ir = popxl.Ir()
main = ir.main_graph

with main, popxl.in_sequence():
    x = np.array(1).astype(np.int32)

    # Create a remote buffer
    buffer = popxl.remote_buffer(x.shape, dtypes.int32, 1)

    # Create a remote variable and locate it to the buffer at index 0
    remote_x = popxl.remote_variable(x, buffer, 0)

    # Load the remote variable
    loaded_x = ops.remote_load(buffer, 0)

    # Calculation on IPU to update the loaded variable
    y = popxl.variable(2)
    ops.var_updates.accumulate_(loaded_x, y)

    # Store the updated value back to the remote buffer
    ops.remote_store(buffer, 0, loaded_x)

    # remote_var end

    y_d2h = popxl.d2h_stream(loaded_x.shape, loaded_x.dtype)
    ops.host_store(y_d2h, loaded_x)

with popxl.Session(ir, "ipu_model") as session:
    outputs = session.run({})
    # Get the data for the remote variable
    final_weight = session.get_tensor_data(remote_x)

assert outputs[y_d2h] == 3
assert final_weight == 3
