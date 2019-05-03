import poponnx
import pytest
import test_util as tu
import numpy as np


def test_virtual_graph():

    poponnx.getLogger().setLevel("TRACE")

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i3 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i4 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    o1 = builder.aiOnnx.add([i1, i2])
    o2 = builder.aiOnnx.add([i3, i4])
    o = builder.aiOnnx.add([o1, o2])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 1)
    builder.virtualGraph(o2, 1)
    builder.virtualGraph(o, 1)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()
    opts.enableVirtualGraphs = True

    s = poponnx.InferenceSession(
        fnModel=proto,
        dataFeed=dataFlow,
        userOptions=opts,
        deviceInfo=tu.get_ipu_model(numIPUs=2))

    s.prepareDevice()

    pass


def test_virtual_graph2():

    poponnx.getLogger().setLevel("TRACE")

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i3 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i4 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))

    with builder.virtualGraph(0):
        o1 = builder.aiOnnx.add([i1, i2])
        o2 = builder.aiOnnx.add([i3, i4])

    with builder.virtualGraph(1):
        o = builder.aiOnnx.add([o1, o2])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    opts = poponnx.SessionOptionsCore()
    opts.enableVirtualGraphs = True

    s = poponnx.InferenceSession(
        fnModel=proto,
        dataFeed=dataFlow,
        userOptions=opts,
        deviceInfo=tu.get_ipu_model(numIPUs=2))

    s.prepareDevice()


def test_virtual_graph3():

    poponnx.getLogger().setLevel("TRACE")

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i3 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i4 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))

    with builder.virtualGraph(3):
        o1 = builder.aiOnnx.add([i1, i2])
        o2 = builder.aiOnnx.add([i3, i4])

    with builder.virtualGraph(2):
        o3 = builder.aiOnnx.add([o1, o2])
        o = builder.aiOnnx.add([i1, o3])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    # Need to anchor the output of the backward pass to stop it being pruned
    dataFlow = poponnx.DataFlow(
        1, {
            o: poponnx.AnchorReturnType("ALL"),
            poponnx.reservedGradientPrefix() + i1:
            poponnx.AnchorReturnType("ALL"),
            poponnx.reservedGradientPrefix() + i2:
            poponnx.AnchorReturnType("ALL"),
            poponnx.reservedGradientPrefix() + i3:
            poponnx.AnchorReturnType("ALL"),
            poponnx.reservedGradientPrefix() + i4:
            poponnx.AnchorReturnType("ALL")
        })

    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]
    #Make sure that the loss is also assigned to a virtual graph
    losses[0].virtualGraph(1)
    optimizer = poponnx.ConstSGD(0.01)

    opts = poponnx.SessionOptionsCore()
    opts.enableVirtualGraphs = True

    s = poponnx.TrainingSession(
        fnModel=proto,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        userOptions=opts,
        deviceInfo=tu.get_ipu_model(numIPUs=4))

    s.prepareDevice()

    anchors = s.initAnchorArrays()

    data1 = np.ones([1], dtype=np.float32)
    data2 = np.ones([1], dtype=np.float32)
    data3 = np.ones([1], dtype=np.float32)
    data4 = np.ones([1], dtype=np.float32)

    inputs = {i1: data1, i2: data2, i3: data3, i4: data4}
    stepio = poponnx.PyStepIO(inputs, anchors)

    s.run(stepio)
    s.weightsFromHost()


def test_virtual_graph4():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i3 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))

    with builder.virtualGraph(3):
        o1 = builder.aiOnnx.add([i1, i2])
        o2 = builder.aiOnnx.add([i3, o1])

    with builder.virtualGraph(2):
        o3 = builder.aiOnnx.mul([i1, i3])

    builder.addOutputTensor(o2)
    builder.addOutputTensor(o3)

    proto = builder.getModelProto()

    # Need to anchor the output of the backward pass to stop it being pruned
    dataFlow = poponnx.DataFlow(
        1, {
            o1: poponnx.AnchorReturnType("ALL"),
            o2: poponnx.AnchorReturnType("ALL"),
            o3: poponnx.AnchorReturnType("ALL"),
            poponnx.reservedGradientPrefix() + i1:
            poponnx.AnchorReturnType("ALL"),
            poponnx.reservedGradientPrefix() + i2:
            poponnx.AnchorReturnType("ALL"),
            poponnx.reservedGradientPrefix() + i3:
            poponnx.AnchorReturnType("ALL")
        })

    losses = [
        poponnx.L1Loss(o1, "l1LossVal_1", 0.1),
        poponnx.L1Loss(o2, "l1LossVal_2", 0.1),
        poponnx.L1Loss(o3, "l1LossVal_3", 0.1)
    ]
    #Make sure that the loss is also assigned to a virtual graph
    losses[0].virtualGraph(3)
    losses[1].virtualGraph(3)
    losses[2].virtualGraph(2)
    optimizer = poponnx.ConstSGD(0.01)

    opts = poponnx.SessionOptionsCore()
    opts.enableVirtualGraphs = True

    s = poponnx.TrainingSession(
        fnModel=proto,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        userOptions=opts,
        deviceInfo=tu.get_ipu_model(numIPUs=4))

    s.prepareDevice()

    anchors = s.initAnchorArrays()

    data1 = np.ones([1], dtype=np.float32)
    data2 = np.ones([1], dtype=np.float32)
    data3 = np.ones([1], dtype=np.float32)

    inputs = {i1: data1, i2: data2, i3: data3}
    stepio = poponnx.PyStepIO(inputs, anchors)

    s.run(stepio)
    s.weightsFromHost()


def test_virtual_graph_bad_index():

    # poponnx.getLogger().setLevel("TRACE")
    #
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
    # opts.enableVirtualGraphs = True
    #
    # s = poponnx.Session(fnModel=proto, dataFeed=dataFlow, userOptions=opts, deviceInfo=tu.get_ipu_model(numIPUs = 2))
    # s.prepareDevice()

    pass
