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
    o = builder.aiOnnx.conv([c1, i2],
                            dilations=[1, 1],
                            pads=[1, 1, 1, 1],
                            strides=[1, 1])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    losses = [popart.L1Loss(o, "l1LossVal", 0.1)]

    opts = popart.SessionOptionsCore()

    session = popart.TrainingSession(
        fnModel=proto,
        dataFeed=popart.DataFlow(stepSize, anchors),
        losses=losses,
        optimizer=optimizer,
        deviceInfo=tu.get_ipu_model(compileIPUCode=False))

    session.prepareDevice()
    session.weightsFromHost()
    session.optimizerFromHost()

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

    lrName = popart.reservedDefaultScaledLearningRate0Prefix() + "FLOAT"
    wdName = popart.reservedDefaultWeightDecayScaleFactor0Prefix() + "FLOAT"
    lsName = popart.reservedLossScalingPrefix() + "FLOAT"

    anchorNames = {
        lrName: popart.AnchorReturnType("ALL"),
        wdName: popart.AnchorReturnType("ALL"),
        lsName: popart.AnchorReturnType("ALL")
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
        session.updateOptimizer(
            popart.SGD({
                "defaultLearningRate": (stepLr, False),
                "defaultWeightDecay": (stepWd, False),
                "lossScaling": (stepLs, False)
            }))
        session.optimizerFromHost()

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
    (learnRate streamed from host), one with a ConstSGD optimizer, othwerwise
    identical.
    We show that if the learning rates match, the training updates are 
    identical, otherwise they differ.
    """
    anchorNames = {"l1LossVal": popart.AnchorReturnType("ALL")}
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

        # set scalar learnRate
        inputsUserSgd["learningRate_FLOAT"] = np.ones(stepSize,
                                                      dtype=np.float32) * lr
        stepioUserSgd = popart.PyStepIO(inputsUserSgd, anchorsArraysUserSgd)

        if step == numSteps - 1:
            sessionUserSgd.updateOptimizer(
                popart.SGD({
                    "defaultLearningRate": (2 * lr, False),
                    "defaultWeightDecay": (2 * wd, False),
                    "lossScaling": (2 * ls, False)
                }))
            sessionUserSgd.optimizerFromHost()

        sessionConstSgd.run(stepioConstSgd)
        sessionUserSgd.run(stepioUserSgd)

        if step == numSteps - 1:
            # We expect to see the diverging losses on the second forward pass
            # after updating the optimizer
            assert (np.array_equal(anchorsArraysUserSgd["l1LossVal"][0],
                                   anchorsArraysConstSgd["l1LossVal"][0]))
            assert (np.array_equal(anchorsArraysUserSgd["l1LossVal"][1],
                                   anchorsArraysConstSgd["l1LossVal"][1]) is
                    False)
        else:
            assert (np.array_equal(anchorsArraysUserSgd["l1LossVal"],
                                   anchorsArraysConstSgd["l1LossVal"]))


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

    out = c2
    builder.addOutputTensor(out)

    proto = builder.getModelProto()

    optimizer = popart.SGD({
        "defaultLearningRate": (0.1, False),
        "defaultWeightDecay": (0.1, False),
        "lossScaling": (1000, False)
    })
    losses = [popart.L1Loss(out, "l1LossVal", 0.1)]

    anchorNames = {
        popart.reservedGradientPrefix() + inid1:
        popart.AnchorReturnType("ALL"),
    }

    opts = popart.SessionOptionsCore()

    session = popart.TrainingSession(
        fnModel=proto,
        dataFeed=popart.DataFlow(1, anchorNames),
        losses=losses,
        optimizer=optimizer,
        deviceInfo=tu.get_ipu_model(compileIPUCode=False))

    session.prepareDevice()
    session.weightsFromHost()
    session.optimizerFromHost()

    anchorArrays = session.initAnchorArrays()

    stepio = popart.PyStepIO({inid1: input1}, anchorArrays)
    session.run(stepio)
