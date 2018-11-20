import poponnx
import pytest


def test_summary_report_before_execution(tmpdir):

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add("1", poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    earlyInfo.add("2", poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))

    dataFlow = poponnx.DataFlow(1, 1, [], poponnx.AnchorReturnType.ALL)
    optimizer = poponnx.SGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        outputdir=str(tmpdir))

    session.initAnchorArrays()
    session.setDevice("IPU")

    with pytest.raises(poponnx.exception) as e_info:
        session.getSummaryReport()


def test_graph_report_before_execution(tmpdir):

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add("1", poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    earlyInfo.add("2", poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))

    dataFlow = poponnx.DataFlow(1, 1, [], poponnx.AnchorReturnType.ALL)
    optimizer = poponnx.SGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        outputdir=str(tmpdir))

    session.initAnchorArrays()
    session.setDevice("IPU")

    with pytest.raises(poponnx.exception) as e_info:
        session.getGraphReport()


def test_execution_report_before_execution(tmpdir):

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add("1", poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    earlyInfo.add("2", poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))

    dataFlow = poponnx.DataFlow(1, 1, [], poponnx.AnchorReturnType.ALL)
    optimizer = poponnx.SGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        outputdir=str(tmpdir))

    session.initAnchorArrays()
    session.setDevice("IPU")

    with pytest.raises(poponnx.exception) as e_info:
        session.getExecutionReport()


def test_summary_report_with_cpu_device(tmpdir):

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add("1", poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    earlyInfo.add("2", poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))

    dataFlow = poponnx.DataFlow(1, 1, [], poponnx.AnchorReturnType.ALL)
    optimizer = poponnx.SGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        outputdir=str(tmpdir))

    session.initAnchorArrays()
    # TODO - this should be a CPU device
    session.setDevice("IPU")

    session.prepareDevice()

    with pytest.raises(poponnx.exception) as e_info:
        session.getExecutionReport()


def test_graph_report_with_cpu_device(tmpdir):

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add("1", poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    earlyInfo.add("2", poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))

    dataFlow = poponnx.DataFlow(1, 1, [], poponnx.AnchorReturnType.ALL)
    optimizer = poponnx.SGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        outputdir=str(tmpdir))

    session.initAnchorArrays()
    # TODO - this should be a CPU device
    session.setDevice("IPU")

    session.prepareDevice()

    with pytest.raises(poponnx.exception) as e_info:
        session.getExecutionReport()


def test_execution_report_with_cpu_device(tmpdir):

    builder = poponnx.Builder()

    shape = poponnx.TensorInfo("FLOAT", [1, 2, 32, 32])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add("1", shape)
    earlyInfo.add("2", shape)

    dataFlow = poponnx.DataFlow(1, 1, [o], poponnx.AnchorReturnType.ALL)
    optimizer = poponnx.SGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        outputdir=str(tmpdir))

    session.initAnchorArrays()
    # TODO - this should be a CPU device
    session.setDevice("IPU")

    session.prepareDevice()

    with pytest.raises(poponnx.exception) as e_info:
        session.getExecutionReport()
