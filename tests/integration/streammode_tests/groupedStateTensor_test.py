# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart
import numpy as np


def init_weight_data(shape):
    return np.random.normal(0, np.sqrt(1 / shape[0]), shape).astype(np.float16)


def model(comm_group, num_replicas, hidden_size):
    variable_settings = popart.VariableSettings(
        popart.CommGroup(comm_group, 0),
        popart.VariableRetrievalMode.OnePerGroup)

    builder = popart.Builder()
    inpt = builder.addInputTensor(
        popart.TensorInfo("FLOAT16", [1, hidden_size]))

    w1_data = init_weight_data((num_replicas, hidden_size, hidden_size))
    w1 = builder.addInitializedInputTensor(w1_data, variable_settings)
    out = builder.aiOnnx.reducesum([builder.aiOnnx.matmul([inpt, w1])])
    builder.addOutputTensor(out)

    return builder, inpt, out


def init_session(out, num_replicas, optimizer, builder):
    options = popart.SessionOptions()
    options.enableReplicatedGraphs = True
    options.replicatedGraphCount = num_replicas

    df = popart.DataFlow(1, {out: popart.AnchorReturnType("ALL")})
    dm = popart.DeviceManager().acquireAvailableDevice(num_replicas)

    session = popart.TrainingSession(
        builder.getModelProto(),
        dataFlow=df,
        deviceInfo=dm,
        userOptions=options,
        loss=out,
        optimizer=optimizer,
    )

    session.prepareDevice()
    anchors = session.initAnchorArrays()
    session.weightsFromHost()
    return session, anchors


def test_verify_addStateTensor():
    """
    This test stresses the addStateTensor to verify that the addStateTensor doesn't segfault. 
    """
    num_replicas = 2
    hidden_size = 2048
    comm_group = popart.CommGroupType.Ungrouped
    optimizer = popart.Adam({})

    builder, inpt, out = model(comm_group, num_replicas, hidden_size)
    session, anchors = init_session(out, num_replicas, optimizer, builder)

    inpt_vals = np.tile(
        np.ones(hidden_size),
        (num_replicas, 1,
         1))  # shape: (num_replicas, batch_size=1, hidden_size)
    session.run(popart.PyStepIO({inpt: inpt_vals.astype(np.float16)}, anchors))
