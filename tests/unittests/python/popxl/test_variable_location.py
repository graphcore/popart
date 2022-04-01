# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import sys
from pathlib import Path

import numpy as np

import popxl
import popxl.dtypes as dtypes
import popxl.ops as ops

# `import test_util` requires adding to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3] / "integration"))
import test_util as tu


@tu.requires_ipu_model
def test_remote_variable():
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        x = np.array(1).astype(np.int32)
        buffer = popxl.RemoteBuffer(x.shape, dtypes.int32, 1)
        remote_x = popxl.remote_variable(x, buffer, 0)

        y = popxl.variable(2)

        loaded_x = ops.remote_load(buffer, 0)
        updated_x = loaded_x + y

        ops.remote_store(buffer, 0, updated_x)

        y_d2h = popxl.d2h_stream(updated_x.shape, updated_x.dtype)
        ops.host_store(y_d2h, updated_x)

    result, stored = run(ir, y_d2h, remote_x)
    assert result == 3
    assert stored == 3


@tu.requires_ipu_model
def test_remote_variable_replica_not_sharded():
    ir = popxl.Ir()
    ir.replication_factor = 2
    ir.num_host_transfers = 1
    main = ir.main_graph

    with main:
        x = np.array(1).astype(np.int32)
        buffer = popxl.RemoteBuffer(x.shape, dtypes.int32, 1)
        remote_x = popxl.remote_variable(x, buffer, 0)

        y = popxl.variable(2, name="y")

        loaded_x = ops.remote_load(buffer, 0)

        updated_x = loaded_x + y  # add x and y

        ops.remote_store(buffer, 0, updated_x)

        y_d2h = popxl.d2h_stream(updated_x.shape, updated_x.dtype)
        ops.host_store(y_d2h, updated_x)

    result, stored = run(ir, y_d2h, remote_x)
    np.testing.assert_equal(result, [3, 3])
    np.testing.assert_equal(stored, [3, 3])


@tu.requires_ipu_model
def test_remote_replia_sharded_variable_gather():
    ir = popxl.Ir()
    ir.replication_factor = 2
    ir.num_host_transfers = 1
    main = ir.main_graph

    with main:
        x = np.array([1, 2]).astype(np.int32)
        buffer = popxl.RemoteBuffer((x.size // 2, ), dtypes.int32, 1)
        remote_x = popxl.remote_replica_sharded_variable(x, buffer, 0)

        y = popxl.variable([3, 4], name="y")

        # loaded_x.shape = (1,)
        loaded_x = ops.remote_load(buffer, 0)
        # full_x.shape = (2,)
        full_x = ops.collectives.replicated_all_gather(loaded_x)

        updated_x = full_x + y

        updated_shard = ops.collectives.replicated_reduce_scatter(
            updated_x, 'local', None, True)
        ops.remote_store(buffer, 0, updated_shard)

        y_d2h = popxl.d2h_stream(updated_x.shape, updated_x.dtype)
        ops.host_store(y_d2h, updated_x)

    result, stored = run(ir, y_d2h, remote_x)
    np.testing.assert_equal(result, [[4, 6], [4, 6]])
    np.testing.assert_equal(stored, [4, 6])


@tu.requires_ipu_model
def test_replia_sharded_variable_gather():
    ir = popxl.Ir()
    opts = ir._pb_ir.getSessionOptions()
    ir.replication_factor = 2
    ir.num_host_transfers = 1
    opts.enableInplaceAmbiguityChecking = False
    main = ir.main_graph

    with main:
        x = np.array([1, 2]).astype(np.int32)
        remote_x, loaded_x = popxl.replica_sharded_variable(x, dtypes.int32)

        y = popxl.variable([3, 4])

        # full_x.shape = (2,)
        full_x = ops.collectives.replicated_all_gather(loaded_x)

        updated_x = ops.scaled_add_(full_x, y)

        updated_shard = ops.collectives.replicated_reduce_scatter(
            updated_x, 'local', None, True)
        # Extra copy_var_update_ required to update `loaded_x`
        ops.var_updates.copy_var_update_(loaded_x, updated_shard)

        y_d2h = popxl.d2h_stream(updated_x.shape, updated_x.dtype)
        ops.host_store(y_d2h, updated_x)

    result, stored = run(ir, y_d2h, remote_x)
    np.testing.assert_equal(result, [[4, 6], [4, 6]])
    np.testing.assert_equal(stored, [4, 6])


@tu.requires_ipu_model
def test_remote_replia_sharded_reuse_buffer():
    ir = popxl.Ir()
    ir.replication_factor = 2
    ir.num_host_transfers = 1
    main = ir.main_graph

    with main:
        x1 = np.array([1, 2]).astype(np.int32)
        x2 = np.array([3, 4]).astype(np.int32)
        buffer = popxl.RemoteBuffer((x1.size // 2, ), dtypes.int32, 2)
        popxl.remote_replica_sharded_variable(x1, buffer, 0, name="x1")
        popxl.remote_replica_sharded_variable(x2, buffer, 1, name="x2")

        ops.remote_load(buffer, 0)
        ops.remote_load(buffer, 1)

    assert buffer.meta_shape == (2, )


def run(ir, out, weight):
    session = popxl.Session(ir, "ipu_model")

    outputs = session.run({})
    session._pb_session.weightsToHost()

    final_weight = session.get_tensor_data(weight)

    return outputs[out], final_weight
