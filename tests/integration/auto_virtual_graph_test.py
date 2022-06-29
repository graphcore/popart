# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest
import test_util as tu
import json


def test_auto_virtual_graph_subgraphs_1():

    ipus = 1

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    input_shape = [1, 64]
    input1 = builder.addInputTensor(popart.TensorInfo("FLOAT16", input_shape))
    input2 = builder.addInputTensor(popart.TensorInfo("FLOAT16", input_shape))

    # Subgraph 0
    w = builder.addInitializedInputTensor(np.zeros([64, 64], np.float16))
    x0 = builder.aiOnnx.matmul([input1, w])
    w = builder.addInitializedInputTensor(np.zeros([64, 64], np.float16))
    x0 = builder.aiOnnx.matmul([x0, w])

    # Subgraph 1
    w = builder.addInitializedInputTensor(np.zeros([64, 64], np.float16))
    x1 = builder.aiOnnx.matmul([input2, w])

    output = x1
    builder.addOutputTensor(output)

    # Only possible split is:
    # ipu1: 0, 1

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {output: popart.AnchorReturnType("Final")})

    opts = popart.SessionOptions()
    opts.virtualGraphMode = popart.VirtualGraphMode.Auto

    with tu.create_test_device(numIpus=ipus) as device:

        session = popart.InferenceSession(fnModel=proto,
                                          dataFlow=dataFlow,
                                          userOptions=opts,
                                          deviceInfo=device)
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))
    for op in ir["maingraph"]:
        print(op)
        assert (int(op["attributes"]["__ipu_number"]) == 0)


def test_auto_virtual_graph_subgraphs_2():

    ipus = 2

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    input_shape = [1, 64]
    input1 = builder.addInputTensor(popart.TensorInfo("FLOAT16", input_shape))
    input2 = builder.addInputTensor(popart.TensorInfo("FLOAT16", input_shape))

    # Subgraph 0
    w = builder.addInitializedInputTensor(np.zeros([64, 64], np.float16),
                                          "TESTID-A")
    x0 = builder.aiOnnx.matmul([input1, w])
    w = builder.addInitializedInputTensor(np.zeros([64, 64], np.float16),
                                          "TESTID-B")
    x0 = builder.aiOnnx.matmul([x0, w])

    # Subgraph 1
    w = builder.addInitializedInputTensor(np.zeros([64, 64], np.float16),
                                          "TESTID-C")
    x1 = builder.aiOnnx.matmul([input2, w])

    # Subgraph 2
    x2 = builder.aiOnnx.add([x0, x1])
    w = builder.addInitializedInputTensor(np.zeros([64, 64], np.float16),
                                          "TESTID-D")
    x2 = builder.aiOnnx.matmul([x2, w])

    output = x2
    builder.addOutputTensor(output)

    # Desired split is:
    # ipu1: 0. ipu2: 1,2

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {output: popart.AnchorReturnType("Final")})

    opts = popart.SessionOptions()
    opts.virtualGraphMode = popart.VirtualGraphMode.Auto

    with tu.create_test_device(numIpus=ipus) as device:

        session = popart.InferenceSession(fnModel=proto,
                                          dataFlow=dataFlow,
                                          userOptions=opts,
                                          deviceInfo=device)
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))
    for op in ir["maingraph"]:
        ipu = op["attributes"]["__ipu_number"]
        for input in op["inputs"]:
            if ("TESTID-A" in input["name"]):
                assert (int(ipu) == 0)
            if ("TESTID-B" in input["name"]):
                assert (int(ipu) == 0)
            if ("TESTID-C" in input["name"]):
                assert (int(ipu) == 1)
            if ("TESTID-D" in input["name"]):
                assert (int(ipu) == 1)


def test_auto_virtual_graph_subgraphs_4():

    ipus = 4

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    input_shape = [1, 64]
    input1 = builder.addInputTensor(popart.TensorInfo("FLOAT16", input_shape))
    input2 = builder.addInputTensor(popart.TensorInfo("FLOAT16", input_shape))

    # Subgraph 0
    w = builder.addInitializedInputTensor(np.zeros([64, 64], np.float16),
                                          "TESTID-A")
    x0 = builder.aiOnnx.matmul([input1, w])
    w = builder.addInitializedInputTensor(np.zeros([64, 64], np.float16),
                                          "TESTID-B")
    x0 = builder.aiOnnx.matmul([x0, w])

    # Subgraph 1
    w = builder.addInitializedInputTensor(np.zeros([64, 64], np.float16),
                                          "TESTID-C")
    x1 = builder.aiOnnx.matmul([input2, w])

    # Subgraph 2
    x2 = builder.aiOnnx.add([x0, x1])
    w = builder.addInitializedInputTensor(np.zeros([64, 64], np.float16),
                                          "TESTID-D")
    x2 = builder.aiOnnx.matmul([x2, w])

    output = x2
    builder.addOutputTensor(output)

    # Desired split is:
    # ipu1: 0, ipu2: 0.5, ipu3: 1, ipu4: 2

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {output: popart.AnchorReturnType("Final")})

    opts = popart.SessionOptions()
    opts.virtualGraphMode = popart.VirtualGraphMode.Auto

    with tu.create_test_device(numIpus=ipus) as device:

        session = popart.InferenceSession(fnModel=proto,
                                          dataFlow=dataFlow,
                                          userOptions=opts,
                                          deviceInfo=device)
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))
    for op in ir["maingraph"]:
        ipu = op["attributes"]["__ipu_number"]
        for input in op["inputs"]:
            if ("TESTID-A" in input["name"]):
                assert (int(ipu) == 0)
            if ("TESTID-B" in input["name"]):
                assert (int(ipu) == 1)
            if ("TESTID-C" in input["name"]):
                assert (int(ipu) == 2)
            if ("TESTID-D" in input["name"]):
                assert (int(ipu) == 3)


def test_auto_virtual_graph_inf_2():

    ipus = 2

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    input_shape = [1, 64]
    input = builder.addInputTensor(popart.TensorInfo("FLOAT16", input_shape))

    x = input
    for _ in range(ipus):
        w = builder.addInitializedInputTensor(np.zeros([64, 64], np.float16))
        x = builder.aiOnnx.matmul([x, w])
    output = x
    builder.addOutputTensor(output)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {output: popart.AnchorReturnType("Final")})

    opts = popart.SessionOptions()
    opts.virtualGraphMode = popart.VirtualGraphMode.Auto

    with tu.create_test_device(numIpus=ipus) as device:

        popart.InferenceSession(fnModel=proto,
                                dataFlow=dataFlow,
                                userOptions=opts,
                                deviceInfo=device)


def test_auto_virtual_graph_inf_many():

    ipus = 4

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    input_shape = [1, 64]
    input = builder.addInputTensor(popart.TensorInfo("FLOAT16", input_shape))

    x = input
    for _ in range(16):
        w = builder.addInitializedInputTensor(np.zeros([64, 64], np.float16))
        x = builder.aiOnnx.matmul([x, w])
    output = x
    builder.addOutputTensor(output)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {output: popart.AnchorReturnType("Final")})

    opts = popart.SessionOptions()
    opts.virtualGraphMode = popart.VirtualGraphMode.Auto

    with tu.create_test_device(numIpus=ipus) as device:

        popart.InferenceSession(fnModel=proto,
                                dataFlow=dataFlow,
                                userOptions=opts,
                                deviceInfo=device)


def test_auto_virtual_graph_train():

    ipus = 2

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    input_shape = [1, 64]
    input = builder.addInputTensor(popart.TensorInfo("FLOAT16", input_shape))

    x = input
    for _ in range(ipus):
        w = builder.addInitializedInputTensor(np.zeros([64, 64], np.float16))
        x = builder.aiOnnx.matmul([x, w])
    output = x
    builder.addOutputTensor(output)

    loss = builder.aiGraphcore.identityloss([output])

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {loss: popart.AnchorReturnType("Final")})

    opts = popart.SessionOptions()
    opts.virtualGraphMode = popart.VirtualGraphMode.Auto

    with tu.create_test_device(numIpus=ipus) as device:

        popart.TrainingSession(fnModel=proto,
                               dataFlow=dataFlow,
                               userOptions=opts,
                               loss=loss,
                               optimizer=popart.SGD(
                                   {"defaultLearningRate": (0.01, True)}),
                               deviceInfo=device)


@tu.requires_ipu_model
def test_auto_virtual_graph_not_enough_splits():
    ipus = 4

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    input_shape = [1, 64]
    input = builder.addInputTensor(popart.TensorInfo("FLOAT16", input_shape))

    x = input
    for _ in range(2):
        w = builder.addInitializedInputTensor(np.zeros([64, 64], np.float16))
        x = builder.aiOnnx.matmul([x, w])
    output = x
    builder.addOutputTensor(output)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {output: popart.AnchorReturnType("Final")})

    opts = popart.SessionOptions()
    opts.virtualGraphMode = popart.VirtualGraphMode.Auto

    with tu.create_test_device(numIpus=ipus) as device:

        with pytest.raises(popart.popart_exception) as e_info:
            popart.InferenceSession(fnModel=proto,
                                    dataFlow=dataFlow,
                                    userOptions=opts,
                                    deviceInfo=device)

    assert (e_info.value.args[0].startswith(
        "[AutoVirtualGraph] Couldn't find enough splits"))
