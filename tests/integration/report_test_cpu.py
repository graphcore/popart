# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import popart
import pytest
import test_util as tu


def test_summary_report_with_cpu_device():

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

    session.prepareDevice()

    with pytest.raises(popart.poplar_exception) as e_info:
        session.getSummaryReport()

    assert (e_info.value.args[0].endswith(
        "Profiling is disabled for current device type."))


def test_graph_report_with_cpu_device():

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

    session.prepareDevice()

    with pytest.raises(popart.poplar_exception) as e_info:
        session.getSummaryReport()

    assert (e_info.value.args[0].endswith(
        "Profiling is disabled for current device type."))


def test_execution_report_with_cpu_device():

    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", [1, 2, 32, 32])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    session = popart.InferenceSession(fnModel=proto,
                                      dataFlow=dataFlow,
                                      deviceInfo=tu.create_test_device())

    session.initAnchorArrays()

    session.prepareDevice()

    with pytest.raises(popart.poplar_exception) as e_info:
        session.getSummaryReport()

    assert (e_info.value.args[0].endswith(
        "Profiling is disabled for current device type."))
