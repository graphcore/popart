import numpy as np
import pytest
import poponnx
import torch
import test_util as tu


def test_get_op_types():
    ops_public = poponnx.getSupportedOperations(False)
    assert (len(ops_public) > 0)

    ops_all = poponnx.getSupportedOperations(True)
    assert (len(ops_all) > 0)
    assert (len(ops_all) > len(ops_public))


def test_add(tmpdir):

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

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        outputdir=str(tmpdir))

    session.setDevice(tu.get_poplar_cpu_device())
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    inputs = {
        i1: np.array([1, 2], dtype=np.float32),
        i2: np.array([3, 4], dtype=np.float32)
    }
    stepio = poponnx.PyStepIO(inputs, anchors)

    session.infer(stepio)

    assert (np.array_equal(anchors[o], [4, 6]))


def test_convolution(tmpdir):

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

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        outputdir=str(tmpdir))

    session.setDevice(tu.get_poplar_cpu_device())
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


def test_matmul(tmpdir):
    # create a basic model with a matmul operator
    # and compare the answer against numpy.matmul

    builder = poponnx.Builder()

    i1shape = poponnx.TensorInfo("FLOAT", [2, 3])
    i2shape = poponnx.TensorInfo("FLOAT", [3, 4])

    i1 = builder.addInputTensor(i1shape)
    i2 = builder.addInputTensor(i2shape)
    o = builder.matmul([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add(i1, i1shape)
    earlyInfo.add(i2, i2shape)

    dataFlow = poponnx.DataFlow(1, 1, [o], poponnx.AnchorReturnType.ALL)

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        outputdir=str(tmpdir))

    session.setDevice(tu.get_poplar_cpu_device())
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    inputs = {
        i1:
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        i2:
        np.array([[7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]],
                 dtype=np.float32)
    }

    stepio = poponnx.PyStepIO(inputs, anchors)

    session.infer(stepio)

    # test the poponnx answer against the numpy answer
    assert (np.array_equal(anchors[o], np.matmul(inputs[i1], inputs[i2])))


def test_averagepool(tmpdir):
    builder = poponnx.Builder()

    data_shape = poponnx.TensorInfo("FLOAT", [1, 1, 6, 6])
    i1 = builder.addInputTensor(data_shape)

    o = builder.averagepool([i1], [2, 2], [2, 2], [0, 0, 0, 0])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add(i1, data_shape)

    dataFlow = poponnx.DataFlow(1, 1, [o], poponnx.AnchorReturnType.ALL)

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        outputdir=str(tmpdir))

    session.setDevice(tu.get_poplar_cpu_device())
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    data = np.random.rand(1, 1, 6, 6).astype(np.float32)

    inputs = {i1: data}

    stepio = poponnx.PyStepIO(inputs, anchors)
    session.infer(stepio)

    # get the pytorch output
    torch_avgpool = torch.nn.AvgPool2d(2, 2)
    torch_output = torch_avgpool(torch.from_numpy(data))

    assert (np.array_equal(torch_output.numpy(), anchors[o]))


def test_maxpool(tmpdir):
    builder = poponnx.Builder()

    data_shape = poponnx.TensorInfo("FLOAT", [1, 1, 6, 6])
    i1 = builder.addInputTensor(data_shape)

    o = builder.maxpool([i1], [2, 2], [2, 2], [0, 0, 0, 0])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add(i1, data_shape)

    dataFlow = poponnx.DataFlow(1, 1, [o], poponnx.AnchorReturnType.ALL)

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        outputdir=str(tmpdir))

    session.setDevice(tu.get_poplar_cpu_device())
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    data = np.random.rand(1, 1, 6, 6).astype(np.float32)

    inputs = {i1: data}

    stepio = poponnx.PyStepIO(inputs, anchors)
    session.infer(stepio)

    # get the pytorch output
    torch_avgpool = torch.nn.MaxPool2d(2, 2)
    torch_output = torch_avgpool(torch.from_numpy(data))

    assert (np.array_equal(torch_output.numpy(), anchors[o]))
