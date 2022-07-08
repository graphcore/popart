# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""
The example shows how to use a remote RTS variable.
"""

import numpy as np

import popxl
import popxl.dtypes as dtypes
import popxl.ops as ops

# remote_var begin
ir = popxl.Ir()
ir.replication_factor = 2
ir.num_host_transfers = 1

with ir.main_graph, popxl.in_sequence():
    # Create an RTS variable remote_x with buffer
    x = np.array([1, 2]).astype(np.int32)
    buffer = popxl.remote_buffer((x.size // 2,), dtypes.int32, 1)
    remote_x = popxl.remote_replica_sharded_variable(x, buffer, 0)
    # Load remote_x to loaded_x
    loaded_x = ops.remote_load(buffer, 0)

    # Create a variable y
    y = popxl.variable([3, 4])

    # Add y to the all gathered full x
    full_x = ops.collectives.replicated_all_gather(loaded_x)
    ops.var_updates.accumulate_(full_x, y)

    # Scatter the updated full x to each buffer across replicas
    updated_shard = ops.collectives.replica_sharded_slice(full_x)
    ops.remote_store(buffer, 0, updated_shard)

# Execute the ir
with popxl.Session(ir, "ipu_model") as session:
    outputs = session.run({})
    # Get the updated x value
    final_weight = session.get_tensor_data(remote_x)

# remote_var end
print(final_weight)
assert np.allclose(final_weight, [4, 6])
