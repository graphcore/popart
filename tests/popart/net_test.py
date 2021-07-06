# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
import pytest

import popart
import test_util as tu


def test_net_from_string():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    popart.InferenceSession(fnModel=proto,
                            dataFlow=dataFlow,
                            deviceInfo=tu.create_test_device())


def test_net_from_file():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    with open("test.onnx", "wb") as f:
        f.write(proto)

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    popart.InferenceSession(fnModel="test.onnx",
                            dataFlow=dataFlow,
                            deviceInfo=tu.create_test_device())


def test_net_failure1():
    # Anchor tensor required in inference mode

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {})

    with pytest.raises(popart.popart_exception) as e_info:
        popart.InferenceSession(fnModel=proto,
                                dataFlow=dataFlow,
                                deviceInfo=tu.create_test_device())

    assert (e_info.type == popart.popart_exception)
    assert (e_info.value.args[0] ==
            "User must specify an anchor tensor when doing inference.")


def test_net_failure2():

    dataFlow = popart.DataFlow(1, {})

    with pytest.raises(popart.popart_exception) as e_info:
        popart.InferenceSession(fnModel="nothing",
                                dataFlow=dataFlow,
                                deviceInfo=tu.create_test_device())

    assert (e_info.type == popart.popart_exception)
    assert (e_info.value.args[0].startswith(
        "Failed to load a ModelProto from the string"))
