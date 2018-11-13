import pytest

import poponnx


def test_net_from_string():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add(i1, i2)
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add("1", poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    earlyInfo.add("2", poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))

    dataFlow = poponnx.DataFlow(1, 1, [], poponnx.AnchorReturnType.ALL)
    optimizer = poponnx.SGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        outputdir="/tmp")


def test_net_from_file():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add(i1, i2)
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    with open("test.onnx", "wb") as f:
        f.write(proto)

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add("1", poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    earlyInfo.add("2", poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))

    dataFlow = poponnx.DataFlow(1, 1, [], poponnx.AnchorReturnType.ALL)
    optimizer = poponnx.SGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    poponnx.Session(
        fnModel="test.onnx",
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        outputdir="/tmp")


def test_net_failure():

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add("1", poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    earlyInfo.add("2", poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))

    dataFlow = poponnx.DataFlow(1, 1, [], poponnx.AnchorReturnType.ALL)
    optimizer = poponnx.SGD(0.01)
    losses = [poponnx.L1Loss("None", "l1LossVal", 0.1)]

    with pytest.raises(poponnx.exception) as e_info:
        poponnx.Session(
            fnModel="nothing",
            earlyInfo=earlyInfo,
            dataFeed=dataFlow,
            losses=losses,
            optimizer=optimizer,
            outputdir="/tmp")

    assert (e_info.type == poponnx.exception)
    assert (e_info.value.args[0] == "Failed to parse ModelProto from string")
