import numpy as np
import poponnx
import pytest
import test_util as tu


def test_ipu_copy_bca1():

    poponnx.getLogger().setLevel("TRACE")

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    o1 = builder.aiOnnx.add([i1, i2])
    o2 = builder.aiOnnx.add([i1, i2])
    o = builder.aiOnnx.add([o1, o2])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 0)
    builder.virtualGraph(o2, 0)
    builder.virtualGraph(o, 1)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()
    opts.enableVirtualGraphs = True

    s = poponnx.Session(fnModel=proto, dataFeed=dataFlow, userOptions=opts)
    s.setDevice(tu.get_ipu_model(numIPUs=3))
    s.prepareDevice()


# Will fail due to an invalid virtual graph
def test_ipu_copy_aca1():

    poponnx.getLogger().setLevel("TRACE")

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    o1 = builder.aiOnnx.add([i1, i2])
    o2 = builder.aiOnnx.add([i1, i2])
    o = builder.aiOnnx.add([o1, o2])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 0)
    builder.virtualGraph(o2, 0)
    builder.virtualGraph(o, 10)  # << Invalid virtual graph

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()
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

    poponnx.getLogger().setLevel("TRACE")

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    o1 = builder.aiOnnx.add([i1, i2])
    o2 = builder.aiOnnx.add([i1, i2])
    t1 = builder.aiOnnx.transpose([i1], [])
    o3 = builder.aiOnnx.add([o1, o2])
    o = builder.aiOnnx.add([o3, t1])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 0)
    builder.virtualGraph(o2, 2)
    builder.virtualGraph(t1, 2)
    builder.virtualGraph(o3, 1)
    builder.virtualGraph(o, 1)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()
    opts.enableVirtualGraphs = True

    s = poponnx.Session(fnModel=proto, dataFeed=dataFlow, userOptions=opts)
    s.setDevice(tu.get_ipu_model(numIPUs=3))
    s.prepareDevice()


# Test to ensure that same tensor it not copied multiple times to the same IPU
def test_ipu_copy_bca2():

    poponnx.getLogger().setLevel("TRACE")

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    o1 = builder.aiOnnx.add([i1, i2])
    o2 = builder.aiOnnx.add([i1, i2])

    o3 = builder.aiOnnx.add([o1, o2])
    o4 = builder.aiOnnx.add([o1, o2])

    o = builder.aiOnnx.add([o3, o4])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 0)
    builder.virtualGraph(o2, 0)
    builder.virtualGraph(o3, 1)
    builder.virtualGraph(o4, 1)

    builder.virtualGraph(o, 2)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()
    opts.enableVirtualGraphs = True

    s = poponnx.Session(fnModel=proto, dataFeed=dataFlow, userOptions=opts)
    s.setDevice(tu.get_ipu_model(numIPUs=3))
    s.prepareDevice()


# Test to make sure that if a single op has multiple it mapped to multiple inputs then the copy does
# the right thing
def test_ipu_copy_bca3():

    poponnx.getLogger().setLevel("TRACE")

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    o1 = builder.aiOnnx.add([i1, i2])
    o = builder.aiOnnx.add([o1, o1])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 0)
    builder.virtualGraph(o, 1)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()
    opts.enableVirtualGraphs = True

    s = poponnx.Session(fnModel=proto, dataFeed=dataFlow, userOptions=opts)
    s.setDevice(tu.get_ipu_model(numIPUs=2))
    s.prepareDevice()


def test_ipu_copy_bca5():

    poponnx.getLogger().setLevel("TRACE")

    builder = poponnx.Builder()

    constData = np.random.rand(2, 2).astype(np.float32)
    c1 = builder.aiOnnx.constant(constData, "constShapeData")
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [2, 2]))
    o1 = builder.aiOnnx.add([c1, i2])
    o2 = builder.aiOnnx.add([c1, i2])
    t1 = builder.aiOnnx.transpose([c1], [])
    o3 = builder.aiOnnx.add([o1, o2])
    o = builder.aiOnnx.add([o3, t1])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 0)
    builder.virtualGraph(o2, 2)
    builder.virtualGraph(t1, 2)
    builder.virtualGraph(o3, 1)
    builder.virtualGraph(o, 1)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()
    opts.enableVirtualGraphs = True

    s = poponnx.Session(fnModel=proto, dataFeed=dataFlow, userOptions=opts)
    s.setDevice(tu.get_ipu_model(numIPUs=3))
    s.prepareDevice()
