# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart
import numpy as np
import test_util as tu
import pytest


@pytest.mark.parametrize(
    "optimizer_name", ("ADAM_NO_BIAS", "ADAM_WITH_BIAS", "SGD_SEPARATE", "SGD_COMBINED")
)
def test_replica_sharded_with_optimizer(optimizer_name):
    np.random.seed(0)

    builder = popart.Builder()
    # Define number of replicas
    num_ipus = 2

    # 1) DEFINE THE MODEL
    # Different tensor of length 3 living on each device
    w_value = np.random.rand(num_ipus, 3).astype(np.float16)
    settings = popart.VariableSettings(
        popart.CommGroup(popart.CommGroupType.Ungrouped, 0),
        popart.VariableRetrievalMode.OnePerGroup,
    )
    w = builder.addInitializedInputTensor(w_value, settings, "w")

    # Dummy loss
    loss = builder.aiOnnx.reducesumsquare([w], debugContext="L2NormLoss")

    # 2) DEFINE ANCHORS
    anchor_desc = {
        w: popart.AnchorReturnType("ALL"),
        loss: popart.AnchorReturnType("ALL"),
    }
    steps_per_execution = 1
    dataFlow = popart.DataFlow(steps_per_execution, anchor_desc)

    # 3) DEFINE THE TRAINING SESSION
    userOpts = popart.SessionOptions()
    userOpts.enableReplicatedGraphs = True
    userOpts.replicatedGraphCount = num_ipus

    # Define the optimizer
    optimizer_settings = {}
    values_adam = {
        "defaultLearningRate": (0.1, True)
    }  # dummy learning rate value to define the optimizer
    values_sgd = {
        "defaultLearningRate": (0.1, True),
        "defaultMomentum": (0.9, True),
    }  # include momentum for SGD
    optimizer_settings["ADAM_NO_BIAS"] = popart.Adam(
        values_adam, mode=popart.AdamMode.AdamNoBias
    )
    optimizer_settings["ADAM_WITH_BIAS"] = popart.Adam(
        values_adam, mode=popart.AdamMode.Adam
    )
    optimizer_settings["SGD_SEPARATE"] = popart.SGD(
        values_sgd, accumulatorAndMomentum=popart.SGDAccumulatorAndMomentum.Separate
    )
    optimizer_settings["SGD_COMBINED"] = popart.SGD(
        values_sgd, accumulatorAndMomentum=popart.SGDAccumulatorAndMomentum.Combined
    )
    optimizer = optimizer_settings[optimizer_name]

    # 4) GET THE DEVICES
    with tu.create_test_device(numIpus=num_ipus) as device:
        session = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            loss=loss,
            deviceInfo=device,
            optimizer=optimizer,
            dataFlow=dataFlow,
            userOptions=userOpts,
        )
        session.prepareDevice()
        anchors = session.initAnchorArrays()

        # 5) RUN THE TRAINING LOOP
        num_steps = 1
        session.weightsFromHost()
        for i in range(num_steps):
            stepio = popart.PyStepIO({}, anchors)
            session.run(stepio, f"Step {i}")
            # Check that we were able to run
            assert anchors[w].shape == w_value.shape
