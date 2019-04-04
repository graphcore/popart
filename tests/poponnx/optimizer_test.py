import numpy as np
import pytest
import poponnx
import test_util as tu


def trainSession(anchors, optimizer, stepSize):

    poponnx.getLogger().setLevel("TRACE")

    builder = poponnx.Builder()

    dataShape = poponnx.TensorInfo("FLOAT", [1, 2, 4, 4])
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

    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    opts = poponnx.SessionOptionsCore()

    session = poponnx.TrainingSession(
        fnModel=proto,
        dataFeed=poponnx.DataFlow(stepSize, anchors),
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
    anchorNames = {
        "learnRate_FLOAT": poponnx.AnchorReturnType("ALL"),
        "weightDecay_FLOAT": poponnx.AnchorReturnType("ALL")
    }

    # Just a placeholder optimizer. We overwrite the hyper-parameters in this
    # test once the session is created
    userSgd = poponnx.SGD(learning_rate=-1, weight_decay=-1)
    stepSize = 2

    session, inputsUserSgd = trainSession(anchorNames, userSgd, stepSize)
    anchorsArrays = session.initAnchorArrays()

    # train
    numSteps = 3
    learningRate = np.random.rand(numSteps).astype('float32')
    weightDecay = np.random.rand(numSteps).astype('float32')

    for step in range(numSteps):

        # Update learning rate parameter between training steps
        stepLr = learningRate[step]
        stepWd = weightDecay[step]
        session.updateOptimizer(
            poponnx.SGD(learning_rate=stepLr, weight_decay=stepWd))
        session.optimizerFromHost()

        stepio = poponnx.PyStepIO(inputsUserSgd, anchorsArrays)

        session.run(stepio)

        assert (np.array_equal(anchorsArrays["learnRate_FLOAT"][0], stepLr))
        # The weight decay tensor is scaled by lr on the host
        # before training
        assert (np.array_equal(anchorsArrays["weightDecay_FLOAT"][0],
                               stepWd * stepLr))


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
    wd = 0.01
    stepSize = 2

    constSgd = poponnx.ConstSGD(learning_rate=lr, weight_decay=wd)
    sessionConstSgd, inputsConstSgd = trainSession(anchorNames, constSgd,
                                                   stepSize)
    anchorsArraysConstSgd = sessionConstSgd.initAnchorArrays()

    userSgd = poponnx.SGD(learning_rate=lr, weight_decay=wd)
    sessionUserSgd, inputsUserSgd = trainSession(anchorNames, userSgd,
                                                 stepSize)
    anchorsArraysUserSgd = sessionUserSgd.initAnchorArrays()

    # train
    numSteps = 3
    for step in range(numSteps):

        stepioConstSgd = poponnx.PyStepIO(inputsConstSgd,
                                          anchorsArraysConstSgd)

        # set scalar learnRate
        inputsUserSgd["learningRate_FLOAT"] = np.ones(
            stepSize, dtype=np.float32) * lr
        stepioUserSgd = poponnx.PyStepIO(inputsUserSgd, anchorsArraysUserSgd)

        if step == numSteps - 1:
            sessionUserSgd.updateOptimizer(
                poponnx.SGD(learning_rate=2 * lr, weight_decay=2 * wd))
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
    poponnx.getLogger().setLevel("TRACE")

    input1 = np.zeros((2, 2, 4, 4), dtype=np.float16)
    input2 = np.zeros((2, 2, 3, 3), dtype=np.float16)
    input3 = np.zeros((2, 2, 3, 3), dtype=np.float16)

    builder = poponnx.Builder()
    inid1 = builder.addInputTensor(poponnx.TensorInfo(input1))
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

    optimizer = poponnx.SGD(learning_rate=0.1, weight_decay=0.1)
    losses = [poponnx.L1Loss(out, "l1LossVal", 0.1)]

    anchorNames = {
        'd__' + inid1: poponnx.AnchorReturnType("ALL"),
    }

    opts = poponnx.SessionOptionsCore()

    session = poponnx.TrainingSession(
        fnModel=proto,
        dataFeed=poponnx.DataFlow(1, anchorNames),
        losses=losses,
        optimizer=optimizer,
        deviceInfo=tu.get_ipu_model(compileIPUCode=False))

    session.prepareDevice()
    session.weightsFromHost()
    session.optimizerFromHost()

    anchorArrays = session.initAnchorArrays()

    stepio = poponnx.PyStepIO({inid1: input1}, anchorArrays)
    session.run(stepio)
