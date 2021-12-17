# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import sys
from pathlib import Path

import numpy as np

import popart
import popart.ir as pir
import popart.ir.ops as ops

# `import test_util` requires adding to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3] / "integration"))
import test_util as tu


@tu.requires_ipu_model
def test_remote_variable():
    ir = pir.Ir()
    main = ir.main_graph()

    with main:
        x = pir.variable(1, name="x")
        buffer = pir.RemoteBuffer(ir, x.shape, x.dtype, 1)
        remote_x = pir.remote_variable(x, buffer, 0)

        y = pir.variable(2)

        loaded_x = ops.remote_load(buffer, remote_x)

        updated_x = loaded_x + y

        ops.remote_store(buffer, remote_x, updated_x)

        y_d2h = pir.d2h_stream(x.shape, x.dtype)
        ops.host_store(y_d2h, updated_x)

    result, stored = run(ir, y_d2h, x)
    assert result == 3
    assert stored == 3


@tu.requires_ipu_model
def test_remote_replia_sharded_variable_gather():
    ir = pir.Ir()
    opts = ir._pb_ir.getSessionOptions()
    opts.enableReplicatedGraphs = True
    opts.replicatedGraphCount = 2
    main = ir.main_graph()

    with main:
        x = pir.variable([1, 2], name="x")
        buffer = pir.RemoteBuffer(ir, (x.nelms // 2, ), x.dtype, 1)
        remote_x = pir.remote_replica_sharded_variable(x, buffer, 0)

        y = pir.variable([3, 4])

        # loaded_x.shape = (1,)
        loaded_x = ops.remote_load(buffer, remote_x)
        # full_x.shape = (2,)
        full_x = ops.collectives.replicated_all_gather(loaded_x)

        updated_x = full_x + y

        updated_shard = ops.collectives.replicated_reduce_scatter(
            updated_x, 'local', None, True)
        ops.remote_store(buffer, remote_x, updated_shard)

        y_d2h = pir.d2h_stream(updated_x.shape, updated_x.dtype)
        ops.host_store(y_d2h, updated_x)

    result, stored = run(ir, y_d2h, x)
    np.testing.assert_equal(result, [[4, 6], [4, 6]])
    np.testing.assert_equal(stored, [4, 6])


@tu.requires_ipu_model
def test_replia_sharded_variable_gather():
    ir = pir.Ir()
    opts = ir._pb_ir.getSessionOptions()
    opts.enableReplicatedGraphs = True
    opts.replicatedGraphCount = 2
    main = ir.main_graph()

    with main:
        x = pir.variable([1, 2], name="x")
        buffer = pir.RemoteBuffer(ir, (x.nelms // 2, ), x.dtype, 1)
        remote_x, loaded_x = pir.replica_sharded_variable(x, buffer, 0)

        y = pir.variable([3, 4])

        # full_x.shape = (2,)
        full_x = ops.collectives.replicated_all_gather(loaded_x)

        updated_x = ops.scaled_add_(full_x, y)

        updated_shard = ops.collectives.replicated_reduce_scatter(
            updated_x, 'local', None, True)
        # Extra copy_var_update_ required to update `loaded_x`
        ops.var_updates.copy_var_update_(loaded_x, updated_shard)

        y_d2h = pir.d2h_stream(updated_x.shape, updated_x.dtype)
        ops.host_store(y_d2h, updated_x)

    result, stored = run(ir, y_d2h, x)
    np.testing.assert_equal(result, [[4, 6], [4, 6]])
    np.testing.assert_equal(stored, [4, 6])


@tu.requires_ipu_model
def test_replica_sharded_variable_no_gather():
    ir = pir.Ir()
    opts = ir._pb_ir.getSessionOptions()
    opts.enableReplicatedGraphs = True
    opts.replicatedGraphCount = 2
    main = ir.main_graph()

    with main:
        x = pir.variable([1, 2], name="x")
        buffer = pir.RemoteBuffer(ir, (x.nelms // 2, ), x.dtype, 1)
        remote_x, loaded_x = pir.replica_sharded_variable(x, buffer, 0)

        y = pir.variable([3, 4])
        sharded_y = ops.collectives.replicated_reduce_scatter(
            y, 'local', None, True)

        # Add to sharded x
        updated_x = ops.scaled_add_(loaded_x, sharded_y)

        y_d2h = pir.d2h_stream(updated_x.shape, updated_x.dtype)
        ops.host_store(y_d2h, updated_x)

    result, stored = run(ir, y_d2h, x)
    print(result.flatten())
    assert np.allclose(result.flatten(), [4, 6])
    assert np.allclose(stored, [4, 6])


@tu.requires_ipu_model
def test_remote_replia_sharded_variable_no_gather():
    ir = pir.Ir()
    opts = ir._pb_ir.getSessionOptions()
    opts.enableReplicatedGraphs = True
    opts.replicatedGraphCount = 2
    main = ir.main_graph()

    with main:
        x = pir.variable([1, 2], name="x")
        buffer = pir.RemoteBuffer(ir, (x.nelms // 2, ), x.dtype, 1)
        remote_x = pir.remote_replica_sharded_variable(x, buffer, 0)

        y = pir.variable([3, 4])
        sharded_y = ops.collectives.replicated_reduce_scatter(
            y, 'local', None, True)

        # loaded_x.shape = (1,)
        loaded_x = ops.remote_load(buffer, remote_x, "x")

        # Add to sharded x
        updated_x = ops.scaled_add_(loaded_x, sharded_y)

        ops.remote_store(buffer, remote_x, updated_x)

        y_d2h = pir.d2h_stream(updated_x.shape, updated_x.dtype)
        ops.host_store(y_d2h, updated_x)

    result, stored = run(ir, y_d2h, x)
    np.testing.assert_equal(result.flatten(), [4, 6])
    np.testing.assert_equal(stored, [4, 6])


def run(ir, out, weight):
    ir = ir._pb_ir
    ir.logIr()
    dataFlow = popart.DataFlow(
        batchesPerStep=1,
        anchorTensors={out.tensor_id(): popart.AnchorReturnType("All")})
    ir.setDataFlow(dataFlow)

    opts = ir.getSessionOptions()
    opts.useHostCopyOps = True
    opts.enableExplicitMainLoops = True
    opts.aliasZeroCopy = True
    opts.explicitRecomputation = True

    ir.updateVertices()

    session = popart.InferenceSession.fromIr(ir=ir,
                                             deviceInfo=tu.create_test_device(
                                                 opts.replicatedGraphCount))

    session.prepareDevice()

    # Create buffers for anchors
    anchors = session.initAnchorArrays()

    # Run the model
    stepio = popart.PyStepIO(inputs={}, outputs=anchors)

    session.weightsFromHost()
    session.run(stepio)
    session.weightsToHost()
    final_weight = np.zeros(weight.shape, weight.dtype.as_numpy())
    session.readWeights(popart.PyWeightsIO({weight.id: final_weight}))

    return anchors[out.tensor_id()], final_weight
