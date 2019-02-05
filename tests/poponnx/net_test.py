import pytest

import poponnx


def test_net_from_string(tmpdir):

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    poponnx.Session(fnModel=proto, dataFeed=dataFlow)


def test_net_from_file(tmpdir):

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    with open("test.onnx", "wb") as f:
        f.write(proto)

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    poponnx.Session(fnModel="test.onnx", dataFeed=dataFlow)


def test_net_failure1(tmpdir):
    # Anchor tensor required in inference and eval modes

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {})

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        poponnx.Session(fnModel=proto, dataFeed=dataFlow)

    assert (e_info.type == poponnx.poponnx_exception)
    assert (
        e_info.value.args[0] ==
        "User must specify an anchor tensor when doing inference or evalulation."
    )


def test_net_failure2(tmpdir):

    dataFlow = poponnx.DataFlow(1, {})

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        poponnx.Session(fnModel="nothing", dataFeed=dataFlow)

    assert (e_info.type == poponnx.poponnx_exception)
    assert (e_info.value.args[0] == "Failed to parse ModelProto from string")
