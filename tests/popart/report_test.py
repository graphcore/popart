# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest
import test_util as tu


@tu.requires_ipu_model
def test_summary_report_before_execution(tmpdir):

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
def test_graph_report_before_execution(tmpdir):

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
        session.getGraphReport()

    assert (e_info.value.args[0].endswith(
        "Session must have been prepared before a report can be fetched"))


@tu.requires_ipu_model
def test_execution_report_before_execution(tmpdir):

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
        session.getExecutionReport()

    assert (e_info.value.args[0].endswith(
        "Session must have been prepared before a report can be fetched"))


@tu.requires_ipu_model
def test_compilation_report(tmpdir):

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

    assert (len(session.getGraphReport()) > 0)


@tu.requires_ipu_model
def test_compilation_report_cbor(tmpdir):

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

    assert (len(session.getGraphReport(True)) > 0)


@tu.requires_ipu_model
def test_execution_report(tmpdir):

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

    rep = session.getExecutionReport()


@tu.requires_ipu_model
def test_execution_report_reset(tmpdir):

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

    rep1 = session.getExecutionReport(resetProfile=False)
    rep2 = session.getExecutionReport(resetProfile=False)
    assert len(rep1) == len(rep2)


@tu.requires_ipu_model
def test_execution_report_cbor(tmpdir):

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

    rep = session.getExecutionReport(True)


@tu.requires_ipu_model
def test_tensor_tile_mapping(tmpdir):

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

    m = session.getTensorTileMap()

    assert (len(m) == 3)
    assert (sorted(list(m.keys())) == sorted([i1, i2, o]))

    # Assume a 1216 tile device, and mapping a scalar will put it on tile 0
    assert (len(m[o]) == 1216)

    # There should only be one tile with a non zero interval
    non_zero_intervals = [(tile, i) for tile, i in enumerate(m[o])
                          if len(i) > 0]
    assert len(non_zero_intervals) == 1

    tile, intervals = non_zero_intervals[0]
    assert intervals[0] == (0, 1)


@tu.requires_ipu_model
def test_no_compile(tmpdir):

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
        session.getGraphReport()

    assert (e_info.value.args[0].endswith(
        "Session must have been prepared before a report can be fetched"))


@tu.requires_ipu_model
def test_serialized_graph_report(tmpdir):

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
