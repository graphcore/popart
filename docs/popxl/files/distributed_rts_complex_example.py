# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl
import popxl.ops as ops

# Code Begin
ir = popxl.Ir(replication="popdist")

# Should be 16 in our scenario.
rf = ir.replication_factor
rg = ir.replica_grouping(group_size=4, stride=4)
# 4
num_groups = rg.num_groups
# Shard over half the group, so in this case 2
shard_over = num_groups // 2
# Let (32,) be shape of unsharded v, and we need 4 groups worth of data
v_h = np.arange(0, num_groups * 32).reshape((num_groups, 32))

with ir.main_graph:
    v_handle, v_shard = popxl.replica_sharded_variable(
        v_h, replica_grouping=rg, shard_over=shard_over
    )

    collective_rg = ir.replica_grouping(
        group_size=rg.group_size // shard_over, stride=rg.stride
    )
    v_gathered = ops.collectives.replicated_all_gather(v_shard, collective_rg)

    v_gathered += 1

    v_updated_shard = ops.collectives.replica_sharded_slice(v_gathered, collective_rg)

    ops.var_updates.copy_var_update_(v_shard, v_updated_shard)

with popxl.Session(ir, "ipu_hw") as sess:
    sess.run()
    print(sess.get_tensor_data(v_handle))
    # Will print the full updated tensor reconstructed from all shards.
    # This will be equivalent to `v_h + 1`.

# Code End
