# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import popart.ir as pir
import popart.ir.ops as ops
from popart.ir import dtypes


def test_random_seed_setup():
    ir = pir.Ir()
    main = ir.main_graph
    with main:
        seed_h2d = pir.h2d_stream(shape=(2, ),
                                  dtype=dtypes.uint32,
                                  name='seed_stream')
        seed = ops.host_load(seed_h2d, 'seed')

        x = pir.variable(0.0)
        x = ops.dropout(x, seed + 1, p=0.1)
        y = ops.dropout(x, seed + 2, p=0.7)

        y_d2h = pir.d2h_stream(y.shape, y.dtype, name="y_stream")
        ops.host_store(y_d2h, y)

    replicas = 4
    parent_seed = 1984
    seed_tensors = pir.create_seeds(parent_seed, replicas=replicas)

    ## Run the program
    ir = ir._pb_ir  # Internal ir

    y_id = y_d2h.tensor_id

    dataFlow = popart.DataFlow(
        batchesPerStep=1, anchorTensors={y_id: popart.AnchorReturnType("All")})
    ir.setDataFlow(dataFlow)

    opts = ir.getSessionOptions()
    opts.useHostCopyOps = True
    opts.enableExplicitMainLoops = True
    opts.aliasZeroCopy = True
    opts.explicitRecomputation = True
    opts.enableReplicatedGraphs = True
    opts.replicatedGraphCount = replicas

    ir.updateVertices()

    device = popart.DeviceManager().createIpuModelDevice({"numIPUs": replicas})
    session = popart.InferenceSession.fromIr(ir=ir, deviceInfo=device)

    session.prepareDevice()

    # Create buffers for anchors
    anchors = session.initAnchorArrays()

    # Run the model
    stepio = popart.PyStepIO(inputs={seed_h2d.tensor_id: seed_tensors},
                             outputs=anchors)
    session.weightsFromHost()
    session.run(stepio)
