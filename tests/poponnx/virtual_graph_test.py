import poponnx
import pytest
import test_util as tu


def test_virtual_graph():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i3 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i4 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    o1 = builder.add([i1, i2])
    o2 = builder.add([i3, i4])
    o = builder.add([o1, o2])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 1)
    builder.virtualGraph(o2, 1)
    builder.virtualGraph(o, 1)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()
    opts.logging = {'all': 'TRACE'}
    opts.enableVirtualGraphs = True

    s = poponnx.Session(fnModel=proto, dataFeed=dataFlow, userOptions=opts)
    s.setDevice(tu.get_ipu_model(numIPUs=2))
    s.prepareDevice()

    pass


def test_virtual_graph_bad_index():

    # builder = poponnx.Builder()
    #
    # i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    # i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    # o1 = builder.add([i1, i2])
    # o2 = builder.add([i1, i2])
    # o = builder.add([o1, o2])
    # builder.addOutputTensor(o)
    #
    # builder.virtualGraph(o1, 0)
    # builder.virtualGraph(o2, 1)
    # builder.virtualGraph(o, 2)
    #
    # proto = builder.getModelProto()
    #
    # dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})
    #
    # opts = poponnx.SessionOptionsCore()
    # opts.logging = {'all': 'TRACE'}
    # opts.enableVirtualGraphs = True
    #
    # s = poponnx.Session(fnModel=proto, dataFeed=dataFlow, userOptions=opts)
    # s.setDevice(tu.get_ipu_model(numIPUs = 2))
    # s.prepareDevice()

    pass
