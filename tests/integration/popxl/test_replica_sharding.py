# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popxl
from popxl import ops


def repeat(x: np.ndarray, n: int, axis: int = 0) -> np.ndarray:
    """Repeat array along new axis inserted at position `axis`"""
    return np.repeat(np.expand_dims(x, axis), n, axis=axis)


def test_rts_sharding_factor():
    """Tests using RTS over a subset of the total data parallel replicas"""
    ir = popxl.Ir(replication=8)
    w_rts_grouping = ir.replica_grouping(group_size=4)
    m_rts_grouping = ir.replica_grouping(group_size=2)

    input_x = np.random.normal(0, 1, (4, 4)).astype(np.float32)
    input_w = np.random.normal(0, 1, (4, 4)).astype(np.float32)
    input_m = np.random.normal(0, 1, (4, 4)).astype(np.float32)

    with ir.main_graph, popxl.in_sequence():
        h2d = popxl.h2d_stream((4, 4), popxl.float32)
        x = ops.host_load(h2d)

        # TODO: Test with replica_grouping with stride > 1
        w_v, w_shard = popxl.replica_sharded_variable(
            input_w, popxl.float32, "w", shard_over=w_rts_grouping.group_size
        )

        m_buffer = popxl.replica_sharded_buffer(
            input_m.shape, popxl.float32, shard_over=m_rts_grouping.group_size
        )
        m_v = popxl.remote_replica_sharded_variable(input_m, m_buffer)

        w = ops.collectives.replicated_all_gather(w_shard, group=w_rts_grouping)

        y = x @ w

        m_shard = ops.remote_load(m_buffer, 0)
        m = ops.collectives.replicated_all_gather(m_shard, group=m_rts_grouping)

        z = y @ m

        d2h = popxl.d2h_stream(z.shape, z.dtype)
        ops.host_store(d2h, z)

    input_full = repeat(input_x, ir.replication_factor, axis=0)
    with popxl.Session(ir, "ipu_hw") as session:
        out = session.run({h2d: input_full})[d2h]

    np.testing.assert_almost_equal(out[0], (input_x @ input_w) @ input_m, 6)
    for t in out[1:]:
        np.testing.assert_almost_equal(out[0], t)

    w_data = session.get_tensor_data(w_v)
    np.testing.assert_almost_equal(w_data, input_w)
    m_data = session.get_tensor_data(m_v)
    np.testing.assert_almost_equal(m_data, input_m)


def test_invalid_sharding_factor():
    ir = popxl.Ir(replication=8)
    with ir.main_graph:
        # shard_over higher than ir.replication_factor
        with pytest.raises(ValueError):
            _ = popxl.replica_sharded_buffer((4, 4), popxl.float32, shard_over=16)
        # shard_over higher than replica_grouping.group_size
        with pytest.raises(ValueError):
            _ = popxl.replica_sharded_buffer(
                (4, 4),
                popxl.float32,
                replica_grouping=ir.replica_grouping(group_size=4),
                shard_over=8,
            )
        # replica_grouping.group_size not divisible by shard_over
        with pytest.raises(ValueError):
            _ = popxl.replica_sharded_buffer(
                (4, 4),
                popxl.float32,
                replica_grouping=ir.replica_grouping(group_size=4),
                shard_over=3,
            )


def test_sharding_valid_when_global_replication_but_no_local_replication():
    global_rf = 8

    ir = popxl.Ir()

    # Emulate what state PopDist would set
    opts = ir._pb_ir.getSessionOptions()
    opts.enableDistributedReplicatedGraphs = True
    opts.globalReplicaOffset = 0
    opts.globalReplicationFactor = global_rf

    # Should not throw.
    with ir.main_graph:
        _ = popxl.replica_sharded_buffer((4, 4), popxl.float32, shard_over=global_rf)
