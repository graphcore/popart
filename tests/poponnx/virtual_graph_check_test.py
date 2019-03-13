import poponnx
import pytest
import test_util as tu


def test_no_virtual_graph():

    poponnx.getLogger().setLevel("TRACE")

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    o1 = builder.aiOnnx.add([i1, i2])
    o2 = builder.aiOnnx.add([i1, i2])
    o = builder.aiOnnx.add([o1, o2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()

    poponnx.InferenceSession(
        fnModel=proto, dataFeed=dataFlow, userOptions=opts)


def test_all_virtual_graph():

    poponnx.getLogger().setLevel("TRACE")

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    o1 = builder.aiOnnx.add([i1, i2])
    o2 = builder.aiOnnx.add([i1, i2])
    o = builder.aiOnnx.add([o1, o2])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 0)
    builder.virtualGraph(o2, 1)
    builder.virtualGraph(o, 1)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()

    poponnx.InferenceSession(
        fnModel=proto, dataFeed=dataFlow, userOptions=opts)


def test_mixed_virtual_graph():

    poponnx.getLogger().setLevel("TRACE")

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    o1 = builder.aiOnnx.add([i1, i2])
    o2 = builder.aiOnnx.add([i1, i2])
    o = builder.aiOnnx.add([o1, o2])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 0)
    builder.virtualGraph(o2, 1)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        poponnx.InferenceSession(
            fnModel=proto, dataFeed=dataFlow, userOptions=opts)

    assert (e_info.value.args[0].startswith(
        ("Op(ai.onnx.Add:7, outputs=[{}]) has virtual graph attribute but "
         "Op(ai.onnx.Add:7, outputs=[{}]) does not").format(o1, o)))
