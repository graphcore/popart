import poponnx
import pytest
import test_util as tu


def test_no_virtual_graph():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    o1 = builder.add([i1, i2])
    o2 = builder.add([i1, i2])
    o = builder.add([o1, o2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()
    opts.logging = {'all': 'TRACE'}

    poponnx.Session(fnModel=proto, dataFeed=dataFlow, userOptions=opts)


def test_all_virtual_graph():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    o1 = builder.add([i1, i2])
    o2 = builder.add([i1, i2])
    o = builder.add([o1, o2])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 0)
    builder.virtualGraph(o2, 1)
    builder.virtualGraph(o, 1)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()
    opts.logging = {'all': 'TRACE'}

    poponnx.Session(fnModel=proto, dataFeed=dataFlow, userOptions=opts)


def test_mixed_virtual_graph():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    o1 = builder.add([i1, i2])
    o2 = builder.add([i1, i2])
    o = builder.add([o1, o2])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 0)
    builder.virtualGraph(o2, 1)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()
    opts.logging = {'all': 'TRACE'}

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        poponnx.Session(fnModel=proto, dataFeed=dataFlow, userOptions=opts)

    assert (e_info.value.args[0].startswith(
        "Operation 100(ai.onnx.Add:9) has different virtual graph attribute "
        "to 102(ai.onnx.Add:9)"))
