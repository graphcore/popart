import numpy as np
import poponnx
import pytest
import test_util as tu


def test_ipu_copy_bca1():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    o1 = builder.add([i1, i2])
    o2 = builder.add([i1, i2])
    o = builder.add([o1, o2])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 0)
    builder.virtualGraph(o2, 0)
    builder.virtualGraph(o, 1)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()
    opts.logging = {'all': 'TRACE'}
    opts.enableVirtualGraphs = True

    s = poponnx.Session(fnModel=proto, dataFeed=dataFlow, userOptions=opts)
    s.setDevice(tu.get_ipu_model(numIPUs=3))
    s.prepareDevice()


# Will fail due to an invalid virtual graph
def test_ipu_copy_aca1():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    o1 = builder.add([i1, i2])
    o2 = builder.add([i1, i2])
    o = builder.add([o1, o2])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 0)
    builder.virtualGraph(o2, 0)
    builder.virtualGraph(o, 10)  # << Invalid virtual graph

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()
    opts.logging = {'all': 'TRACE'}
    opts.enableVirtualGraphs = True

    s = poponnx.Session(fnModel=proto, dataFeed=dataFlow, userOptions=opts)
    s.setDevice(tu.get_ipu_model(numIPUs=3))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        s.prepareDevice()

    assert (e_info.value.args[0].startswith(
        "Op(ai.onnx.Add:7, outputs=[{}]) has been assigned to an invalid virtual graph 10"
        .format(o)))


# Test that an input stream tensor is correctly mapped to multiple ipus
def test_ipu_copy_bca4():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    o1 = builder.add([i1, i2])
    o2 = builder.add([i1, i2])
    t1 = builder.transpose([i1], [])
    o3 = builder.add([o1, o2])
    o = builder.add([o3, t1])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 0)
    builder.virtualGraph(o2, 2)
    builder.virtualGraph(t1, 2)
    builder.virtualGraph(o3, 1)
    builder.virtualGraph(o, 1)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()
    opts.logging = {'all': 'TRACE'}
    opts.enableVirtualGraphs = True

    s = poponnx.Session(fnModel=proto, dataFeed=dataFlow, userOptions=opts)
    s.setDevice(tu.get_ipu_model(numIPUs=3))
    s.prepareDevice()


# Test to ensure that same tensor it not copied multiple times to the same IPU
def test_ipu_copy_bca2():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    o1 = builder.add([i1, i2])
    o2 = builder.add([i1, i2])

    o3 = builder.add([o1, o2])
    o4 = builder.add([o1, o2])

    o = builder.add([o3, o4])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 0)
    builder.virtualGraph(o2, 0)
    builder.virtualGraph(o3, 1)
    builder.virtualGraph(o4, 1)

    builder.virtualGraph(o, 2)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()
    opts.logging = {'all': 'TRACE'}
    opts.enableVirtualGraphs = True

    s = poponnx.Session(fnModel=proto, dataFeed=dataFlow, userOptions=opts)
    s.setDevice(tu.get_ipu_model(numIPUs=3))
    s.prepareDevice()


# Test to make sure that if a single op has multiple it mapped to multiple inputs then the copy does
# the right thing
def test_ipu_copy_bca3():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    o1 = builder.add([i1, i2])
    o = builder.add([o1, o1])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 0)
    builder.virtualGraph(o, 1)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()
    opts.logging = {'all': 'TRACE'}
    opts.enableVirtualGraphs = True

    s = poponnx.Session(fnModel=proto, dataFeed=dataFlow, userOptions=opts)
    s.setDevice(tu.get_ipu_model(numIPUs=2))
    s.prepareDevice()


def test_ipu_copy_bca5():
    builder = poponnx.Builder()

    constData = np.random.rand(2, 2).astype(np.float32)
    c1 = builder.constant(constData, "constShapeData")
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [2, 2]))
    o1 = builder.add([c1, i2])
    o2 = builder.add([c1, i2])
    t1 = builder.transpose([c1], [])
    o3 = builder.add([o1, o2])
    o = builder.add([o3, t1])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 0)
    builder.virtualGraph(o2, 2)
    builder.virtualGraph(t1, 2)
    builder.virtualGraph(o3, 1)
    builder.virtualGraph(o, 1)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()
    opts.logging = {'all': 'TRACE'}
    opts.enableVirtualGraphs = True

    s = poponnx.Session(fnModel=proto, dataFeed=dataFlow, userOptions=opts)
    s.setDevice(tu.get_ipu_model(numIPUs=3))
    s.prepareDevice()
