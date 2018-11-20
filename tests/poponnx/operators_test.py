import numpy as np
import pytest
import poponnx


def test_add():

    builder = poponnx.Builder()

    shape = poponnx.TensorInfo("FLOAT", [2])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add(i1, shape)
    earlyInfo.add(i2, shape)

    dataFlow = poponnx.DataFlow(1, 1, [o], poponnx.AnchorReturnType.ALL)
    optimizer = poponnx.SGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        outputdir="/tmp")

    session.setDevice("IPU")
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    inputs = {
        i1: np.array([1, 2], dtype=np.float32),
        i2: np.array([3, 4], dtype=np.float32)
    }
    stepio = poponnx.PyStepIO(inputs, anchors)

    session.infer(stepio)

    assert (np.array_equal(anchors[o], [4, 6]))


def test_convolution():

    builder = poponnx.Builder()

    data_shape = poponnx.TensorInfo("FLOAT", [1, 2, 4, 4])
    filt_shape = poponnx.TensorInfo("FLOAT", [3, 2, 3, 3])

    i1 = builder.addInputTensor(data_shape)
    i2 = builder.addInputTensor(filt_shape)
    o = builder.convolution([i1, i2], [1, 1], [1, 1, 1, 1], [1, 1], 1)
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add(i1, data_shape)
    earlyInfo.add(i2, filt_shape)

    dataFlow = poponnx.DataFlow(1, 1, [o], poponnx.AnchorReturnType.ALL)
    optimizer = poponnx.SGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        outputdir="/tmp")

    session.setDevice("IPU")
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    data = np.ones(data_shape.shape(), dtype=np.float32)
    filt = np.ones(filt_shape.shape(), dtype=np.float32)

    inputs = {i1: data, i2: filt}
    stepio = poponnx.PyStepIO(inputs, anchors)

    session.infer(stepio)

    expected = np.array([[[[8., 12., 12., 8.], [12., 18., 18., 12.],
                           [12., 18., 18., 12.], [8., 12., 12., 8.]],
                          [[8., 12., 12., 8.], [12., 18., 18., 12.],
                           [12., 18., 18., 12.], [8., 12., 12., 8.]],
                          [[8., 12., 12., 8.], [12., 18., 18., 12.],
                           [12., 18., 18., 12.], [8., 12., 12., 8.]]]],
                        dtype=np.float32)

    assert (np.array_equal(anchors[o], expected))
