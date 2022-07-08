# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import popart
import pytest
import test_util as tu
import numpy as np


@tu.requires_ipu_model
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

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    with tu.create_test_device(numIpus=2) as device:
        s = popart.InferenceSession(
            fnModel=proto, dataFlow=dataFlow, userOptions=opts, deviceInfo=device
        )

        s.prepareDevice()


@tu.requires_ipu_model
@pytest.mark.parametrize("container", [set, list])
def test_virtual_graph_multi_output(container):
    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [4, 1]))
    o1, o2, o3, o4 = builder.aiOnnx.split([i1], 4)
    o5 = builder.aiOnnx.add([o1, o2])
    o6 = builder.aiOnnx.add([o3, o4])
    o = builder.aiOnnx.add([o5, o6])
    builder.addOutputTensor(o)

    builder.virtualGraph(container([o1, o2, o3, o4]), 1)
    builder.virtualGraph(o5, 0)
    builder.virtualGraph(o6, 0)
    builder.virtualGraph(o, 0)

    # Check the virtual graph was set correctly
    assert builder.getVirtualGraph(container([o1, o2, o3, o4])) == 1

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    with tu.create_test_device(numIpus=2) as device:
        s = popart.InferenceSession(
            fnModel=proto, dataFlow=dataFlow, userOptions=opts, deviceInfo=device
        )

        s.prepareDevice()


@tu.requires_ipu_model
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

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    with tu.create_test_device(numIpus=2) as device:
        s = popart.InferenceSession(
            fnModel=proto, dataFlow=dataFlow, userOptions=opts, deviceInfo=device
        )

        s.prepareDevice()


@tu.requires_ipu_model
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
        o = builder.aiGraphcore.l1loss([o], 0.1)

    proto = builder.getModelProto()

    # Need to anchor the output of the backward pass to stop it being pruned
    dataFlow = popart.DataFlow(
        1,
        {
            o: popart.AnchorReturnType("All"),
            popart.reservedGradientPrefix() + i1: popart.AnchorReturnType("All"),
            popart.reservedGradientPrefix() + i2: popart.AnchorReturnType("All"),
            popart.reservedGradientPrefix() + i3: popart.AnchorReturnType("All"),
            popart.reservedGradientPrefix() + i4: popart.AnchorReturnType("All"),
        },
    )

    optimizer = popart.SGD({"defaultLearningRate": (0.01, True)})

    opts = popart.SessionOptions()
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    with tu.create_test_device(numIpus=4) as device:
        s = popart.TrainingSession(
            fnModel=proto,
            dataFlow=dataFlow,
            loss=o,
            optimizer=optimizer,
            userOptions=opts,
            deviceInfo=device,
        )

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


@tu.requires_ipu_model
def test_virtual_graph4():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    i3 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))

    with builder.virtualGraph(3):
        o1 = builder.aiOnnx.add([i1, i2])
        o1l1 = builder.aiGraphcore.l1loss([o1], 0.1)
        o2 = builder.aiOnnx.add([i3, o1])
        o2l1 = builder.aiGraphcore.l1loss([o2], 0.1)

    with builder.virtualGraph(2):
        o3 = builder.aiOnnx.mul([i1, i3])
        o3l1 = builder.aiGraphcore.l1loss([o3], 0.1)

    with builder.virtualGraph(3):
        loss = builder.aiOnnx.sum([o1l1, o2l1, o3l1])

    proto = builder.getModelProto()

    # Need to anchor the output of the backward pass to stop it being pruned
    dataFlow = popart.DataFlow(
        1,
        {
            o1: popart.AnchorReturnType("All"),
            o2: popart.AnchorReturnType("All"),
            o3: popart.AnchorReturnType("All"),
            popart.reservedGradientPrefix() + i1: popart.AnchorReturnType("All"),
            popart.reservedGradientPrefix() + i2: popart.AnchorReturnType("All"),
            popart.reservedGradientPrefix() + i3: popart.AnchorReturnType("All"),
        },
    )

    optimizer = popart.ConstSGD(0.01)

    opts = popart.SessionOptions()
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    with tu.create_test_device(numIpus=4) as device:
        s = popart.TrainingSession(
            fnModel=proto,
            dataFlow=dataFlow,
            loss=loss,
            optimizer=optimizer,
            userOptions=opts,
            deviceInfo=device,
        )

        s.prepareDevice()

        anchors = s.initAnchorArrays()

        data1 = np.ones([1], dtype=np.float32)
        data2 = np.ones([1], dtype=np.float32)
        data3 = np.ones([1], dtype=np.float32)

        inputs = {i1: data1, i2: data2, i3: data3}
        stepio = popart.PyStepIO(inputs, anchors)

        s.run(stepio)
        s.weightsFromHost()


@tu.requires_ipu_model
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
    # dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})
    #
    # opts = popart.SessionOptions()
    # opts.virtualGraphMode = popart.VirtualGraphMode.Manual
    #
    # s = popart.TrainingSession(fnModel=proto, dataFlow=dataFlow, userOptions=opts, deviceInfo=tu.create_test_device(numIpus = 2))
    # s.prepareDevice()

    pass


@tu.requires_ipu_model
def test_streaming_optimizer_tensors():
    bps = 5
    input_data = np.random.rand(bps, 2, 2).astype(np.float32)
    weight_data_0 = np.random.rand(2, 2).astype(np.float32)
    weight_data_1 = np.random.rand(2, 2).astype(np.float32)
    weight_data_2 = np.random.rand(2, 2).astype(np.float32)

    def run_test(enablePipelining):
        popart.getLogger().setLevel("TRACE")

        builder = popart.Builder()

        i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", input_data.shape[1::]))
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

        o2l1 = builder.aiGraphcore.l1loss([o2], 0.1)
        if enablePipelining:
            builder.virtualGraph(o2l1, 2)

        proto = builder.getModelProto()

        anchorId = popart.reservedDefaultScaledLearningRate0Prefix() + "FLOAT"

        # Need to anchor the output of the backward pass to stop it being pruned
        dataFlow = popart.DataFlow(bps, [anchorId])

        optimizer = popart.SGD({"defaultLearningRate": (1.0, False)})

        opts = popart.SessionOptions()
        if enablePipelining:
            opts.virtualGraphMode = popart.VirtualGraphMode.Manual
        opts.enablePipelining = enablePipelining

        numIPUs = 1
        if enablePipelining:
            numIPUs = 3

        with tu.create_test_device(numIpus=numIPUs) as device:
            session = popart.TrainingSession(
                fnModel=proto,
                dataFlow=dataFlow,
                loss=o2l1,
                optimizer=optimizer,
                userOptions=opts,
                deviceInfo=device,
            )

            session.prepareDevice()

            anchors = session.initAnchorArrays()

            inputs = {i1: input_data}
            stepio = popart.PyStepIO(inputs, anchors)

            session.weightsFromHost()

            # run 2 steps, changing the optimizer halfway through
            result = []
            session.run(stepio)
            result.append(np.copy(anchors[anchorId]))

            session.updateOptimizerFromHost(
                popart.SGD({"defaultLearningRate": (0.5, False)})
            )

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
