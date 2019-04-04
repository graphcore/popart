import numpy as np
import poponnx
import pytest
import test_util as tu


def test_summary_report_before_execution(tmpdir):

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    session = poponnx.InferenceSession(
        fnModel=proto,
        dataFeed=dataFlow,
        deviceInfo=tu.get_poplar_cpu_device())

    session.initAnchorArrays()

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        session.getSummaryReport()

    assert (e_info.value.args[0].endswith(
        "Session must have been prepared before a report can be fetched"))


def test_graph_report_before_execution(tmpdir):

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    session = poponnx.InferenceSession(
        fnModel=proto,
        dataFeed=dataFlow,
        deviceInfo=tu.get_poplar_cpu_device())

    session.initAnchorArrays()

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        session.getGraphReport()

    assert (e_info.value.args[0].endswith(
        "Session must have been prepared before a report can be fetched"))


def test_execution_report_before_execution(tmpdir):

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    session = poponnx.InferenceSession(
        fnModel=proto,
        dataFeed=dataFlow,
        deviceInfo=tu.get_poplar_cpu_device())

    session.initAnchorArrays()

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        session.getExecutionReport()

    assert (e_info.value.args[0].endswith(
        "Session must have been prepared before a report can be fetched"))


def test_summary_report_with_cpu_device(tmpdir):

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    session = poponnx.InferenceSession(
        fnModel=proto,
        dataFeed=dataFlow,
        deviceInfo=tu.get_poplar_cpu_device())

    session.initAnchorArrays()

    session.prepareDevice()

    with pytest.raises(poponnx.poplar_exception) as e_info:
        session.getExecutionReport()

    assert (e_info.value.args[0].endswith(
        "Profiling is disabled for current device type."))


def test_graph_report_with_cpu_device(tmpdir):

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    session = poponnx.InferenceSession(
        fnModel=proto,
        dataFeed=dataFlow,
        deviceInfo=tu.get_poplar_cpu_device())

    session.initAnchorArrays()

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
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    session = poponnx.InferenceSession(
        fnModel=proto,
        dataFeed=dataFlow,
        deviceInfo=tu.get_poplar_cpu_device())

    session.initAnchorArrays()

    session.prepareDevice()

    with pytest.raises(poponnx.poplar_exception) as e_info:
        session.getExecutionReport()

    assert (e_info.value.args[0].endswith(
        "Profiling is disabled for current device type."))


def test_compilation_report(tmpdir):

    builder = poponnx.Builder()

    shape = poponnx.TensorInfo("FLOAT", [1])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    session = poponnx.InferenceSession(
        fnModel=proto,
        dataFeed=dataFlow,
        deviceInfo=tu.get_ipu_model(compileIPUCode=False))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    assert (len(session.getGraphReport()) > 0)


def test_compilation_report_cbor(tmpdir):

    builder = poponnx.Builder()

    shape = poponnx.TensorInfo("FLOAT", [1])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    session = poponnx.InferenceSession(
        fnModel=proto,
        dataFeed=dataFlow,
        deviceInfo=tu.get_ipu_model(compileIPUCode=False))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    assert (len(session.getGraphReport(True)) > 0)


def test_execution_report(tmpdir):

    builder = poponnx.Builder()

    shape = poponnx.TensorInfo("FLOAT", [1])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    session = poponnx.InferenceSession(
        fnModel=proto,
        dataFeed=dataFlow,
        deviceInfo=tu.get_ipu_model(compileIPUCode=False))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    d1 = np.array([10.]).astype(np.float32)
    d2 = np.array([11.]).astype(np.float32)
    stepio = poponnx.PyStepIO({i1: d1, i2: d2}, anchors)

    session.run(stepio)

    rep = session.getExecutionReport()


def test_execution_report_cbor(tmpdir):

    builder = poponnx.Builder()

    shape = poponnx.TensorInfo("FLOAT", [1])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    session = poponnx.InferenceSession(
        fnModel=proto,
        dataFeed=dataFlow,
        deviceInfo=tu.get_ipu_model(compileIPUCode=False))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    d1 = np.array([10.]).astype(np.float32)
    d2 = np.array([11.]).astype(np.float32)
    stepio = poponnx.PyStepIO({i1: d1, i2: d2}, anchors)

    session.run(stepio)

    rep = session.getExecutionReport(True)


def test_tensor_tile_mapping(tmpdir):

    builder = poponnx.Builder()

    shape = poponnx.TensorInfo("FLOAT", [1])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    session = poponnx.InferenceSession(
        fnModel=proto,
        dataFeed=dataFlow,
        deviceInfo=tu.get_ipu_model(compileIPUCode=False))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    m = session.getTensorTileMap()

    assert (len(m) == 3)
    assert (sorted(list(m.keys())) == sorted([i1, i2, o]))

    # Assume a 1216 tile device, and mapping a scalar will put it on tile 0
    assert (len(m[o]) == 1216)

    for i in enumerate(m[o]):
        tile, intervals = i
        if tile == 0:
            assert (len(intervals) == 1)
            assert (intervals[0] == (0, 1))
        else:
            assert (len(intervals) == 0)


def test_no_compile(tmpdir):

    builder = poponnx.Builder()

    shape = poponnx.TensorInfo("FLOAT", [1])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    opts = poponnx.SessionOptions()
    opts.compileEngine = False
    opts.logging = {"devicex": "INFO"}

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    session = poponnx.InferenceSession(
        proto,
        dataFlow,
        userOptions=opts,
        deviceInfo=tu.get_ipu_model(compileIPUCode=False))

    session.prepareDevice()

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        session.getGraphReport()

    assert (e_info.value.args[0].endswith(
        "Session must have been prepared before a report can be fetched"))
