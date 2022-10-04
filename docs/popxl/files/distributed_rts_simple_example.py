# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops

# Code Begin
ir = popxl.Ir(replication="popdist")
with ir.main_graph:
    v_handle, v_shard = popxl.replica_sharded_variable([1, 1])
    # v_shard will be [1] on each replica

    # Dummy RTS forward pass: gather your shards and perform some ops.
    v_gathered = ops.collectives.replicated_all_gather(v_shard)
    w = popxl.variable([2, 2])
    ops.matmul_(w, v_gathered)

    # Dummy RTS backward pass: you have gradients for the whole v,
    # reduce-scatter them then perform the optimiser step on your shard.
    v_grad = popxl.constant([1, 1])
    v_grad_shard = ops.collectives.replicated_reduce_scatter(v_grad)
    v_shard += v_grad_shard
    # v_shard is now [2] on each replica.

with popxl.Session(ir, "ipu_hw") as sess:
    sess.run()
    print(sess.get_tensor_data(v_handle))
    # -> [2, 2]   <- The full updated tensor reconstructed from all shards

# Code End
