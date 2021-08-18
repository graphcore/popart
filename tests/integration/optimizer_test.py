# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import test_util as tu


def trainSession(anchors, optimizer, stepSize):

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    dataShape = popart.TensorInfo("FLOAT", [1, 2, 4, 4])
    i1 = builder.addInputTensor(dataShape)

    filtInit = np.ones([2, 2, 3, 3], dtype=np.float32)
    i2 = builder.addInitializedInputTensor(filtInit)

    c1 = builder.aiOnnx.conv([i1, i2],
                             dilations=[1, 1],
                             pads=[1, 1, 1, 1],
                             strides=[1, 1])
    c2 = builder.aiOnnx.conv([c1, i2],
                             dilations=[1, 1],
                             pads=[1, 1, 1, 1],
                             strides=[1, 1])
    o = builder.aiGraphcore.l1loss([c2], 0.1)

    proto = builder.getModelProto()

    opts = popart.SessionOptions()

    session = popart.TrainingSession(
        fnModel=proto,
        dataFlow=popart.DataFlow(stepSize, anchors),
        loss=o,
        optimizer=optimizer,
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    session.prepareDevice()
    session.weightsFromHost()

    # add step dimension to infeed
    infeedShape = dataShape.shape()
    infeedShape.insert(0, stepSize)
    data = np.ones(infeedShape, dtype=np.float32)
    inputs = {i1: data}

    return session, inputs


def test_sgd_param_check():
    """
    In this test we check that learning rate tensor, returned as an anchor,
    matches the value supplied to the optimizer constructor
    """

    lrName = popart.TensorId(
        popart.reservedDefaultScaledLearningRate0Prefix() + "FLOAT")
    wdName = popart.TensorId(
        popart.reservedDefaultWeightDecayScaleFactor0Prefix() + "FLOAT")
    lsName = popart.TensorId(popart.reservedLossScalingPrefix() + "FLOAT")

    anchorNames = {
        lrName: popart.AnchorReturnType("All"),
        wdName: popart.AnchorReturnType("All"),
        lsName: popart.AnchorReturnType("All")
    }

    # Just a placeholder optimizer. We overwrite the hyper-parameters in this
    # test once the session is created
    userSGD = popart.SGD({
        "defaultLearningRate": (0.5, False),
        "defaultWeightDecay": (0.6, False),
        "lossScaling": (10.0, False)
    })
    stepSize = 2

    session, inputsUserSgd = trainSession(anchorNames, userSGD, stepSize)
    anchorsArrays = session.initAnchorArrays()

    # train
    numSteps = 3
    learningRate = np.random.rand(numSteps).astype('float32')
    weightDecay = np.random.rand(numSteps).astype('float32')
    lossScaling = np.random.rand(numSteps).astype('float32')

    for step in range(numSteps):

        # Update learning rate parameter between training steps
        stepLr = learningRate[step]
        stepWd = weightDecay[step]
        stepLs = lossScaling[step]
        session.updateOptimizerFromHost(
            popart.SGD({
                "defaultLearningRate": (stepLr, False),
                "defaultWeightDecay": (stepWd, False),
                "lossScaling": (stepLs, False)
            }))

        stepio = popart.PyStepIO(inputsUserSgd, anchorsArrays)

        session.run(stepio)

        assert (np.array_equal(anchorsArrays[lsName][0], stepLs))

        scaled = (stepLr / stepLs)
        assert (np.array_equal(anchorsArrays[lrName][0], scaled))

        # The weight decay tensor is scaled by lr on the host
        # before training
        scaled = 1 - (stepWd * stepLr)
        assert (np.allclose(anchorsArrays[wdName][0], scaled))


def test_constsgd_vs_sgd():
    """
    In this test we run training with two sessions - one with an SGD optimizer
    (learnRate streamed from host), one with a ConstSGD optimizer, otherwise
    identical.
    We show that if the learning rates match, the training updates are 
    identical, otherwise they differ.
    """
    anchorNames = {popart.TensorId("L1:0"): popart.AnchorReturnType("All")}
    lr = 0.01
    wd = 0.01
    ls = 1000
    stepSize = 2

    constSgd = popart.SGD({
        "defaultLearningRate": (lr, True),
        "defaultWeightDecay": (wd, True),
        "lossScaling": (ls, True)
    })

    sessionConstSgd, inputsConstSgd = trainSession(anchorNames, constSgd,
                                                   stepSize)
    anchorsArraysConstSgd = sessionConstSgd.initAnchorArrays()

    userSGD = popart.SGD({
        "defaultLearningRate": (lr, False),
        "defaultWeightDecay": (wd, False),
        "lossScaling": (ls, False)
    })

    sessionUserSgd, inputsUserSgd = trainSession(anchorNames, userSGD,
                                                 stepSize)
    anchorsArraysUserSgd = sessionUserSgd.initAnchorArrays()

    # train
    numSteps = 3
    for step in range(numSteps):

        stepioConstSgd = popart.PyStepIO(inputsConstSgd, anchorsArraysConstSgd)

        stepioUserSgd = popart.PyStepIO(inputsUserSgd, anchorsArraysUserSgd)

        if step == numSteps - 1:
            sessionUserSgd.updateOptimizerFromHost(
                popart.SGD({
                    "defaultLearningRate": (2 * lr, False),
                    "defaultWeightDecay": (2 * wd, False),
                    "lossScaling": (2 * ls, False)
                }))

        sessionConstSgd.run(stepioConstSgd)
        sessionUserSgd.run(stepioUserSgd)

        if step == numSteps - 1:
            # We expect to see the diverging losses on the second forward pass
            # after updating the optimizer
            assert (np.array_equal(
                anchorsArraysUserSgd[popart.TensorId("L1:0")][0],
                anchorsArraysConstSgd[popart.TensorId("L1:0")][0]))
            assert (np.array_equal(
                anchorsArraysUserSgd[popart.TensorId("L1:0")][1],
                anchorsArraysConstSgd[popart.TensorId("L1:0")][1]) is False)
        else:
            assert (np.array_equal(
                anchorsArraysUserSgd[popart.TensorId("L1:0")],
                anchorsArraysConstSgd[popart.TensorId("L1:0")]))


def test_sgd_with_float16_model():
    popart.getLogger().setLevel("TRACE")

    input1 = np.zeros((2, 2, 4, 4), dtype=np.float16)
    input2 = np.zeros((2, 2, 3, 3), dtype=np.float16)
    input3 = np.zeros((2, 2, 3, 3), dtype=np.float16)

    builder = popart.Builder()
    inid1 = builder.addInputTensor(popart.TensorInfo(input1))
    inid2 = builder.addInitializedInputTensor(input2)
    inid3 = builder.addInitializedInputTensor(input2)

    c1 = builder.aiOnnx.conv([inid1, inid2],
                             dilations=[1, 1],
                             pads=[1, 1, 1, 1],
                             strides=[1, 1])
    c2 = builder.aiOnnx.conv([c1, inid3],
                             dilations=[1, 1],
                             pads=[1, 1, 1, 1],
                             strides=[1, 1])

    # Reduce to scalar
    out = builder.aiGraphcore.identityloss([c2])

    proto = builder.getModelProto()

    optimizer = popart.SGD({
        "defaultLearningRate": (0.1, False),
        "defaultWeightDecay": (0.1, False),
        "lossScaling": (1000, False)
    })

    anchorNames = {
        popart.TensorId(popart.reservedGradientPrefix() + inid1):
        popart.AnchorReturnType("All"),
    }

    opts = popart.SessionOptions()

    session = popart.TrainingSession(
        fnModel=proto,
        dataFlow=popart.DataFlow(1, anchorNames),
        loss=out,
        optimizer=optimizer,
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    session.prepareDevice()
    session.weightsFromHost()

    anchorArrays = session.initAnchorArrays()

    stepio = popart.PyStepIO({inid1: input1}, anchorArrays)
    session.run(stepio)


def test_sgd_with_zero_learning_rate():
    """
    In this test we check that we can run a training step zero learning rate,
    and that it behaves as expected (i.e. no weight update)
    """

    # Let's start with an optimizer with a variable, non-zero learning rate
    optSettings = {
        "defaultLearningRate": (0.5, False),
        "defaultWeightDecay": (0.6, False),
        "lossScaling": (10.0, False)
    }
    stepSize = 2
    session, inputsUserSgd = trainSession({}, popart.SGD(optSettings),
                                          stepSize)
    anchorsArrays = session.initAnchorArrays()

    # Get the initial weights:
    fn = "init.onnx"
    session.modelToHost(fn)
    wId = popart.TensorId("init_input")
    weights = {wId: np.empty(shape=[2, 2, 3, 3], dtype=np.float32)}
    weightsio = popart.PyWeightsIO(weights)
    session.readWeights(weightsio)
    init_weights = np.copy(weights[wId])

    # Run for a step with non-zero lr, observe that the weights have changed
    stepio = popart.PyStepIO(inputsUserSgd, anchorsArrays)
    session.run(stepio)
    session.weightsToHost()
    session.readWeights(weightsio)
    updated_weights = np.copy(weights[wId])
    assert np.array_equal(init_weights, updated_weights) is False

    # Update optimizer with zero lr, (only valid if variable)
    optSettings["defaultLearningRate"] = (0.0, True)
    with pytest.raises(popart.popart_exception) as e_info:
        session.updateOptimizerFromHost(popart.SGD(optSettings))
    assert e_info.value.args[0].startswith(
        "Constant, zero learning rate in SGD")

    # Run a training step, and confirm the weights haven't updated
    optSettings["defaultLearningRate"] = (0.0, False)
    session.updateOptimizerFromHost(popart.SGD(optSettings))

    session.weightsToHost()
    session.readWeights(weightsio)
    assert np.array_equal(weights[wId], updated_weights)
