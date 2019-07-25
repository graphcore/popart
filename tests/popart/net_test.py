import pytest

import popart
import test_util as tu


def test_net_from_string(tmpdir):

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

    popart.InferenceSession(fnModel=proto,
                             dataFeed=dataFlow,
                             deviceInfo=tu.get_poplar_cpu_device())


def test_net_from_file(tmpdir):

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    with open("test.onnx", "wb") as f:
        f.write(proto)

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

    popart.InferenceSession(fnModel="test.onnx",
                             dataFeed=dataFlow,
                             deviceInfo=tu.get_poplar_cpu_device())


def test_net_failure1(tmpdir):
    # Anchor tensor required in inference and eval modes

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {})

    with pytest.raises(popart.popart_exception) as e_info:
        popart.InferenceSession(fnModel=proto,
                                 dataFeed=dataFlow,
                                 deviceInfo=tu.get_poplar_cpu_device())

    assert (e_info.type == popart.popart_exception)
    assert (
        e_info.value.args[0] ==
        "User must specify an anchor tensor when doing inference or evalulation."
    )


def test_net_failure2(tmpdir):

    dataFlow = popart.DataFlow(1, {})

    with pytest.raises(popart.popart_exception) as e_info:
        popart.InferenceSession(fnModel="nothing",
                                 dataFeed=dataFlow,
                                 deviceInfo=tu.get_poplar_cpu_device())

    assert (e_info.type == popart.popart_exception)
    assert (e_info.value.args[0] == "Failed to parse ModelProto from string")
