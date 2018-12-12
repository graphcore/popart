import poponnx
import pytest
import test_util as tu


def test_summary_report_before_execution(tmpdir):

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {})

    session = poponnx.Session(fnModel=proto, dataFeed=dataFlow)

    session.initAnchorArrays()
    session.setDevice(tu.get_poplar_cpu_device())

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        session.getSummaryReport()

    assert (e_info.value.args[0].endswith(
        "Session must have been prepared before a report can be fetched"))


def test_graph_report_before_execution(tmpdir):

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {})

    session = poponnx.Session(fnModel=proto, dataFeed=dataFlow)

    session.initAnchorArrays()
    session.setDevice(tu.get_poplar_cpu_device())

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        session.getGraphReport()

    assert (e_info.value.args[0].endswith(
        "Session must have been prepared before a report can be fetched"))


def test_execution_report_before_execution(tmpdir):

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {})

    session = poponnx.Session(fnModel=proto, dataFeed=dataFlow)

    session.initAnchorArrays()
    session.setDevice(tu.get_poplar_cpu_device())

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        session.getExecutionReport()

    assert (e_info.value.args[0].endswith(
        "Session must have been prepared before a report can be fetched"))


def test_summary_report_with_cpu_device(tmpdir):

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {})

    session = poponnx.Session(fnModel=proto, dataFeed=dataFlow)

    session.initAnchorArrays()
    session.setDevice(tu.get_poplar_cpu_device())

    session.prepareDevice()

    with pytest.raises(poponnx.poplar_exception) as e_info:
        session.getExecutionReport()

    assert (e_info.value.args[0].endswith(
        "Profiling is disabled for current device type."))


def test_graph_report_with_cpu_device(tmpdir):

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {})

    session = poponnx.Session(fnModel=proto, dataFeed=dataFlow)

    session.initAnchorArrays()
    session.setDevice(tu.get_poplar_cpu_device())

    session.prepareDevice()

    with pytest.raises(poponnx.poplar_exception) as e_info:
        session.getExecutionReport()

    assert (e_info.value.args[0].endswith(
        "Profiling is disabled for current device type."))


def test_execution_report_with_cpu_device(tmpdir):

    builder = poponnx.Builder()

    shape = poponnx.TensorInfo("FLOAT", [1, 2, 32, 32])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    session = poponnx.Session(fnModel=proto, dataFeed=dataFlow)

    session.initAnchorArrays()
    session.setDevice(tu.get_poplar_cpu_device())

    session.prepareDevice()

    with pytest.raises(poponnx.poplar_exception) as e_info:
        session.getExecutionReport()

    assert (e_info.value.args[0].endswith(
        "Profiling is disabled for current device type."))
