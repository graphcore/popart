# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import popart
import pytest
import test_util as tu


def test_no_virtual_graph():

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    o1 = builder.aiOnnx.add([i1, i2])
    o2 = builder.aiOnnx.add([i1, i2])
    o = builder.aiOnnx.add([o1, o2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()

    with tu.create_test_device() as device:
        popart.InferenceSession(
            fnModel=proto, dataFlow=dataFlow, userOptions=opts, deviceInfo=device
        )


def test_all_virtual_graph():

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    o1 = builder.aiOnnx.add([i1, i2])
    o2 = builder.aiOnnx.add([i1, i2])
    o = builder.aiOnnx.add([o1, o2])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 0)
    builder.virtualGraph(o2, 1)
    builder.virtualGraph(o, 1)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    with tu.create_test_device() as device:
        popart.InferenceSession(
            fnModel=proto, dataFlow=dataFlow, userOptions=opts, deviceInfo=device
        )


def test_mixed_virtual_graph():

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    o1 = builder.aiOnnx.add([i1, i2])
    o2 = builder.aiOnnx.add([i1, i2])
    o = builder.aiOnnx.add([o1, o2])
    builder.addOutputTensor(o)

    builder.virtualGraph(o1, 0)
    builder.virtualGraph(o2, 1)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    with tu.create_test_device() as device:
        with pytest.raises(popart.popart_exception) as e_info:
            popart.InferenceSession(
                fnModel=proto, dataFlow=dataFlow, userOptions=opts, deviceInfo=device
            )

        assert e_info.value.args[0].startswith("Either all")
