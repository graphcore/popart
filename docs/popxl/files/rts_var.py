# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
'''
The example shows how to use an RTS variable.
'''

import numpy as np

import popxl
import popxl.dtypes as dtypes
import popxl.ops as ops

# remote_var begin
ir = popxl.Ir()
ir.replication_factor = 2
ir.num_host_transfers = 1

with ir.main_graph, popxl.in_sequence():
    # Create an RTS variable remote_x and its shards loaded_x
    x = np.array([1, 2]).astype(np.int32)
    remote_x, loaded_x = popxl.replica_sharded_variable(x, dtypes.int32)

    # Create a variable and shard it across replicas
    y = popxl.variable([3, 4])
    sharded_y = ops.collectives.replica_sharded_slice(y)

    # Add each shard of y to each shard of x
    ops.var_updates.accumulate_(loaded_x, sharded_y)

# Execute the ir
with popxl.Session(ir, "ipu_model") as session:
    outputs = session.run({})

# Get the updated x value
final_weight = session.get_tensor_data(remote_x)

# remote_var end
print(final_weight)
assert np.allclose(final_weight, [4, 6])
