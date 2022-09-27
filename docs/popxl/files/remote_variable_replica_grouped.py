# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""
This example shows how to use remote variables that are replica grouped.
"""

import numpy as np

import popxl
import popxl.ops as ops
import popxl.dtypes as dtypes

# RemoteVarReplicaGrouped Begin
ir = popxl.Ir(replication=2)

num_groups = 2
v_h = np.arange(0, num_groups * 32).reshape((num_groups, 32))

rg = ir.replica_grouping(group_size=ir.replication_factor // num_groups)

with ir.main_graph, popxl.in_sequence():
    remote_buffer = popxl.remote_buffer((32,), dtypes.int32)
    remote_v = popxl.remote_variable(v_h, remote_buffer, replica_grouping=rg)

    v = ops.remote_load(remote_buffer, 0)

    v += 1

    ops.remote_store(remote_buffer, 0, v)
# RemoteVarReplicaGrouped End

with popxl.Session(ir, "ipu_model") as sess:
    sess.run()

assert np.array_equiv(sess.get_tensor_data(remote_v), v_h + 1)
