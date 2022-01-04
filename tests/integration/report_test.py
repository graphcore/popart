# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest
import test_util as tu
import tempfile
import pva


@tu.requires_ipu_model
def test_summary_report_before_execution():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    session = popart.InferenceSession(fnModel=proto,
                                      dataFlow=dataFlow,
                                      deviceInfo=tu.create_test_device())

    session.initAnchorArrays()

    with pytest.raises(popart.popart_exception) as e_info:
        session.getSummaryReport()

    assert (e_info.value.args[0].endswith(
        "Session must have been prepared before a report can be fetched"))


def test_graph_report_before_execution():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    session = popart.InferenceSession(fnModel=proto,
                                      dataFlow=dataFlow,
                                      deviceInfo=tu.create_test_device())

    session.initAnchorArrays()

    with pytest.raises(popart.popart_exception) as e_info:
        session.getSummaryReport()

    assert (e_info.value.args[0].endswith(
        "Session must have been prepared before a report can be fetched"))


@tu.requires_ipu_model
def test_execution_report_before_execution():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    session = popart.InferenceSession(fnModel=proto,
                                      dataFlow=dataFlow,
                                      deviceInfo=tu.create_test_device())

    session.initAnchorArrays()

    with pytest.raises(popart.popart_exception) as e_info:
        session.getSummaryReport()

    assert (e_info.value.args[0].endswith(
        "Session must have been prepared before a report can be fetched"))


@tu.requires_ipu_model
def test_report_before_execution():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    session = popart.InferenceSession(fnModel=proto,
                                      dataFlow=dataFlow,
                                      deviceInfo=tu.create_test_device())

    session.initAnchorArrays()

    with pytest.raises(popart.popart_exception) as e_info:
        session.getReport()

    assert (e_info.value.args[0].endswith(
        "Session must have been prepared before a report can be fetched"))


@tu.requires_ipu_model
def test_compilation_report():

    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", [1])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    tempDir = tempfile.TemporaryDirectory()
    options = popart.SessionOptions()
    options.engineOptions["autoReport.directory"] = tempDir.name
    options.engineOptions["autoReport.outputGraphProfile"] = "true"

    session = popart.InferenceSession(
        fnModel=proto,
        dataFlow=dataFlow,
        userOptions=options,
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    report = session.getReport()
    assert (report.compilation.target.totalMemory > 0)
    assert (
        report.compilation.target.architecture == pva.IPU.Architecture.Ipu2)


@tu.requires_ipu_model
def test_compilation_report_deprecated():

    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", [1])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    tempDir = tempfile.TemporaryDirectory()
    options = popart.SessionOptions()
    options.engineOptions["autoReport.directory"] = tempDir.name
    options.engineOptions["autoReport.outputGraphProfile"] = "true"

    session = popart.InferenceSession(
        fnModel=proto,
        dataFlow=dataFlow,
        userOptions=options,
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    assert (len(session.getSummaryReport()) > 0)


@tu.requires_ipu_model
def test_compilation_report_cbor():

    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", [1])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    session = popart.InferenceSession(
        fnModel=proto,
        dataFlow=dataFlow,
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    assert (len(session.getSummaryReport(True)) > 0)


@tu.requires_ipu_model
def test_execution_report():

    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", [1])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    tempDir = tempfile.TemporaryDirectory()
    options = popart.SessionOptions()
    options.engineOptions["autoReport.directory"] = tempDir.name
    options.engineOptions["autoReport.all"] = "true"

    session = popart.InferenceSession(
        fnModel=proto,
        dataFlow=dataFlow,
        userOptions=options,
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    d1 = np.array([10.]).astype(np.float32)
    d2 = np.array([11.]).astype(np.float32)
    stepio = popart.PyStepIO({i1: d1, i2: d2}, anchors)

    session.run(stepio, "Test message")

    report = session.getReport()
    assert (report.execution.runs[0].name == "Test message")


@tu.requires_ipu_model
def test_execution_report_new():

    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", [1])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    tempDir = tempfile.TemporaryDirectory()
    options = popart.SessionOptions()
    options.engineOptions["autoReport.directory"] = tempDir.name
    options.engineOptions["autoReport.all"] = "true"

    session = popart.InferenceSession(
        fnModel=proto,
        dataFlow=dataFlow,
        userOptions=options,
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    d1 = np.array([10.]).astype(np.float32)
    d2 = np.array([11.]).astype(np.float32)
    stepio = popart.PyStepIO({i1: d1, i2: d2}, anchors)

    session.run(stepio, "Test message")

    report = session.getReport()
    assert report.execution.runs[0].name == "Test message"


@tu.requires_ipu_model
def test_execution_report_reset():

    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", [1])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()
    opts.engineOptions = {"debug.instrument": "true"}

    session = popart.InferenceSession(
        fnModel=proto,
        dataFlow=dataFlow,
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    d1 = np.array([10.]).astype(np.float32)
    d2 = np.array([11.]).astype(np.float32)
    stepio = popart.PyStepIO({i1: d1, i2: d2}, anchors)

    session.run(stepio)

    rep1 = session.getSummaryReport(resetProfile=False)
    rep2 = session.getSummaryReport(resetProfile=False)
    assert len(rep1) == len(rep2)


@tu.requires_ipu_model
def test_execution_report_cbor():

    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", [1])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    session = popart.InferenceSession(
        fnModel=proto,
        dataFlow=dataFlow,
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    d1 = np.array([10.]).astype(np.float32)
    d2 = np.array([11.]).astype(np.float32)
    stepio = popart.PyStepIO({i1: d1, i2: d2}, anchors)

    session.run(stepio)

    rep = session.getSummaryReport(True)


@tu.requires_ipu_model
def test_no_compile():

    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", [1])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    opts = popart.SessionOptions()
    opts.compileEngine = False

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    session = popart.InferenceSession(
        proto,
        dataFlow,
        userOptions=opts,
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    session.prepareDevice()

    with pytest.raises(popart.popart_exception) as e_info:
        session.getSummaryReport()

    assert (e_info.value.args[0].endswith(
        "Session must have been prepared before a report can be fetched"))


@tu.requires_ipu_model
def test_serialized_graph_report():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [2, 2]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [2, 2]))
    i3 = builder.addInputTensor(popart.TensorInfo("FLOAT", [2, 2]))

    out = builder.aiOnnx.matmul([i1, i2], "ff1")
    out = builder.aiOnnx.matmul([out, i3], "ff2")
    builder.addOutputTensor(out)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {out: popart.AnchorReturnType("All")})

    session = popart.InferenceSession(fnModel=proto,
                                      dataFlow=dataFlow,
                                      deviceInfo=tu.create_test_device())

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    # This is an encoded capnp report - so not easy to decode here
    rep = session.getSerializedGraph()
    assert (len(rep))
