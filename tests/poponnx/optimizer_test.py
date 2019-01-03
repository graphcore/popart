import numpy as np
import pytest
import poponnx
import test_util as tu


def trainSession(anchors, optimizer, stepSize):

    builder = poponnx.Builder()

    dataShape = poponnx.TensorInfo("FLOAT", [1, 2, 4, 4])
    i1 = builder.addInputTensor(dataShape)

    filtInit = np.ones([2, 2, 3, 3], dtype=np.float32)
    i2 = builder.addInitializedInputTensor(filtInit)

    c1 = builder.convolution([i1, i2], [1, 1], [1, 1, 1, 1], [1, 1], 1)
    o = builder.convolution([c1, i2], [1, 1], [1, 1, 1, 1], [1, 1], 1)
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    opts = poponnx.SessionOptionsCore()
    opts.logging = {"all": "TRACE", "session": "WARN"}

    session = poponnx.Session(
        fnModel=proto,
        dataFeed=poponnx.DataFlow(stepSize, anchors),
        losses=losses,
        optimizer=optimizer)

    session.setDevice(tu.get_ipu_model(compileIPUCode=False))

    session.prepareDevice()
    session.weightsFromHost()
    session.optimizerFromHost()

    # add step dimension to infeed
    infeedShape = dataShape.shape()
    infeedShape.insert(0, stepSize)
    data = np.ones(infeedShape, dtype=np.float32)
    inputs = {i1: data}

    return session, inputs


def test_sgd_learnrate_check():
    """
    In this test we check that learning rate tesnor, returned as an anchor,
    matches the value supplied to the optimizer constructor
    """
    anchorNames = {"learnRate": poponnx.AnchorReturnType("ALL")}
    # Just a placeholder optimizer. We overwrite the lr in this test once
    # the session is created
    userSgd = poponnx.SGD(-1)
    stepSize = 2
    session, inputsUserSgd = trainSession(anchorNames, userSgd, stepSize)
    anchorsArrays = session.initAnchorArrays()

    # train
    numSteps = 3
    learningRate = np.random.rand(numSteps).astype('float32')
    for step in range(numSteps):

        # Update learning rate parameter between training steps
        stepLr = learningRate[step]
        session.updateOptimizer(poponnx.SGD(stepLr))
        session.optimizerFromHost()

        stepio = poponnx.PyStepIO(inputsUserSgd, anchorsArrays)

        session.train(stepio)

        print(anchorsArrays["learnRate"])
        print(stepLr)

        assert (np.array_equal(anchorsArrays["learnRate"][0], stepLr))


def test_constsgd_vs_sgd():
    """
    In this test we run training with two sessions - one with an SGD optimizer
    (learnRate streamed from host), one with a ConstSGD optimizer, othwerwise
    identical.
    We show that if the learning rates match, the training updates are 
    identical, otherwise they differ.
    """
    anchorNames = {"l1LossVal": poponnx.AnchorReturnType("ALL")}
    lr = 0.01
    stepSize = 2

    constSgd = poponnx.ConstSGD(lr)
    sessionConstSgd, inputsConstSgd = trainSession(anchorNames, constSgd,
                                                   stepSize)
    anchorsArraysConstSgd = sessionConstSgd.initAnchorArrays()

    userSgd = poponnx.SGD(lr)
    sessionUserSgd, inputsUserSgd = trainSession(anchorNames, userSgd,
                                                 stepSize)
    anchorsArraysUserSgd = sessionUserSgd.initAnchorArrays()

    # train
    numSteps = 3
    for step in range(numSteps):

        stepioConstSgd = poponnx.PyStepIO(inputsConstSgd,
                                          anchorsArraysConstSgd)

        # set scalar learnRate
        inputsUserSgd["learningRate"] = np.ones(
            stepSize, dtype=np.float32) * lr
        stepioUserSgd = poponnx.PyStepIO(inputsUserSgd, anchorsArraysUserSgd)

        if step == numSteps - 1:
            sessionUserSgd.updateOptimizer(poponnx.SGD(2 * lr))
            sessionUserSgd.optimizerFromHost()

        sessionConstSgd.train(stepioConstSgd)
        sessionUserSgd.train(stepioUserSgd)

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
