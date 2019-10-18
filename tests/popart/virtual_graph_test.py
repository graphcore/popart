import popart
import pytest
import test_util as tu
import numpy as np


def test_virtual_graph():

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    i3 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    i4 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    o1 = builder.aiOnnx.add([i1, i2])
    o2 = builder.aiOnnx.add([i3, i4])
    o = builder.aiOnnx.add([o1, o2])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 1)
    builder.virtualGraph(o2, 1)
    builder.virtualGraph(o, 1)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

    opts = popart.SessionOptionsCore()
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    s = popart.InferenceSession(fnModel=proto,
                                dataFeed=dataFlow,
                                userOptions=opts,
                                deviceInfo=tu.get_ipu_model(numIPUs=2))

    s.prepareDevice()

    pass


def test_virtual_graph2():

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    i3 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    i4 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))

    with builder.virtualGraph(0):
        o1 = builder.aiOnnx.add([i1, i2])
        o2 = builder.aiOnnx.add([i3, i4])

    with builder.virtualGraph(1):
        o = builder.aiOnnx.add([o1, o2])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

    opts = popart.SessionOptionsCore()
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    s = popart.InferenceSession(fnModel=proto,
                                dataFeed=dataFlow,
                                userOptions=opts,
                                deviceInfo=tu.get_ipu_model(numIPUs=2))

    s.prepareDevice()


def test_virtual_graph3():

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    i3 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    i4 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))

    with builder.virtualGraph(3):
        o1 = builder.aiOnnx.add([i1, i2])
        o2 = builder.aiOnnx.add([i3, i4])

    with builder.virtualGraph(2):
        o3 = builder.aiOnnx.add([o1, o2])
        o = builder.aiOnnx.add([i1, o3])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    # Need to anchor the output of the backward pass to stop it being pruned
    dataFlow = popart.DataFlow(
        1, {
            o: popart.AnchorReturnType("ALL"),
            popart.reservedGradientPrefix() + i1:
            popart.AnchorReturnType("ALL"),
            popart.reservedGradientPrefix() + i2:
            popart.AnchorReturnType("ALL"),
            popart.reservedGradientPrefix() + i3:
            popart.AnchorReturnType("ALL"),
            popart.reservedGradientPrefix() + i4:
            popart.AnchorReturnType("ALL")
        })

    losses = [popart.L1Loss(o, "l1LossVal", 0.1)]
    #Make sure that the loss is also assigned to a virtual graph
    losses[0].virtualGraph(1)
    optimizer = popart.SGD({"defaultLearningRate": (0.01, True)})

    opts = popart.SessionOptionsCore()
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    s = popart.TrainingSession(fnModel=proto,
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
    stepio = popart.PyStepIO(inputs, anchors)

    s.run(stepio)
    s.weightsFromHost()


def test_virtual_graph4():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    i3 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))

    with builder.virtualGraph(3):
        o1 = builder.aiOnnx.add([i1, i2])
        o2 = builder.aiOnnx.add([i3, o1])

    with builder.virtualGraph(2):
        o3 = builder.aiOnnx.mul([i1, i3])

    builder.addOutputTensor(o2)
    builder.addOutputTensor(o3)

    proto = builder.getModelProto()

    # Need to anchor the output of the backward pass to stop it being pruned
    dataFlow = popart.DataFlow(
        1, {
            o1: popart.AnchorReturnType("ALL"),
            o2: popart.AnchorReturnType("ALL"),
            o3: popart.AnchorReturnType("ALL"),
            popart.reservedGradientPrefix() + i1:
            popart.AnchorReturnType("ALL"),
            popart.reservedGradientPrefix() + i2:
            popart.AnchorReturnType("ALL"),
            popart.reservedGradientPrefix() + i3:
            popart.AnchorReturnType("ALL")
        })

    losses = [
        popart.L1Loss(o1, "l1LossVal_1", 0.1),
        popart.L1Loss(o2, "l1LossVal_2", 0.1),
        popart.L1Loss(o3, "l1LossVal_3", 0.1)
    ]
    #Make sure that the loss is also assigned to a virtual graph
    losses[0].virtualGraph(3)
    losses[1].virtualGraph(3)
    losses[2].virtualGraph(2)
    optimizer = popart.ConstSGD(0.01)

    opts = popart.SessionOptionsCore()
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    s = popart.TrainingSession(fnModel=proto,
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
    stepio = popart.PyStepIO(inputs, anchors)

    s.run(stepio)
    s.weightsFromHost()


def test_virtual_graph_bad_index():

    # popart.getLogger().setLevel("TRACE")
    #
    # builder = popart.Builder()
    #
    # i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    # i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
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
    # dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})
    #
    # opts = popart.SessionOptionsCore()
    # opts.virtualGraphMode = popart.VirtualGraphMode.Manual
    #
    # s = popart.Session(fnModel=proto, dataFeed=dataFlow, userOptions=opts, deviceInfo=tu.get_ipu_model(numIPUs = 2))
    # s.prepareDevice()

    pass


def test_streaming_optimizer_tensors():
    bps = 5
    input_data = np.random.rand(bps, 2, 2).astype(np.float32)
    weight_data_0 = np.random.rand(2, 2).astype(np.float32)
    weight_data_1 = np.random.rand(2, 2).astype(np.float32)
    weight_data_2 = np.random.rand(2, 2).astype(np.float32)

    def run_test(enablePipelining):
        popart.getLogger().setLevel("TRACE")

        builder = popart.Builder()

        i1 = builder.addInputTensor(
            popart.TensorInfo("FLOAT", input_data.shape[1::]))
        w0 = builder.addInitializedInputTensor(weight_data_0)
        w1 = builder.addInitializedInputTensor(weight_data_1)
        w2 = builder.addInitializedInputTensor(weight_data_2)

        o0 = builder.aiOnnx.matmul([i1, w0])
        if enablePipelining:
            builder.virtualGraph(o0, 0)

        o1 = builder.aiOnnx.matmul([o0, w1])
        if enablePipelining:
            builder.virtualGraph(o1, 1)

        o2 = builder.aiOnnx.matmul([o1, w2])
        if enablePipelining:
            builder.virtualGraph(o2, 2)

        o = o2
        builder.addOutputTensor(o)

        proto = builder.getModelProto()

        anchorId = popart.reservedDefaultScaledLearningRate0Prefix() + "FLOAT"

        # Need to anchor the output of the backward pass to stop it being pruned
        dataFlow = popart.DataFlow(bps,
                                   {anchorId: popart.AnchorReturnType("ALL")})

        losses = [popart.L1Loss(o, "l1LossVal", 0.1)]
        # Make sure that the loss is also assigned to a virtual graph
        if enablePipelining:
            losses[0].virtualGraph(2)

        optimizer = popart.SGD({"defaultLearningRate": (1.0, False)})

        opts = popart.SessionOptionsCore()
        if enablePipelining:
            opts.virtualGraphMode = popart.VirtualGraphMode.Manual
        opts.enablePipelining = enablePipelining

        numIPUs = 1
        if enablePipelining:
            numIPUs = 3

        session = popart.TrainingSession(
            fnModel=proto,
            dataFeed=dataFlow,
            losses=losses,
            optimizer=optimizer,
            userOptions=opts,
            deviceInfo=tu.get_ipu_model(numIPUs=numIPUs))

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        inputs = {i1: input_data}
        stepio = popart.PyStepIO(inputs, anchors)

        session.weightsFromHost()
        session.optimizerFromHost()

        # run 2 steps, changing the optimizer halfway through
        result = []
        session.run(stepio)
        result.append(np.copy(anchors[anchorId]))

        session.updateOptimizer(
            popart.SGD({"defaultLearningRate": (0.5, False)}))
        session.optimizerFromHost()

        session.run(stepio)
        result.append(np.copy(anchors[anchorId]))

        return result

    x = run_test(False)
    y = run_test(True)

    assert len(x) == len(y)

    for i in range(len(x)):
        print(x[i])
        print(y[i])
        print()

        assert np.array_equal(x[i], y[i])
