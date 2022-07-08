# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import test_util as tu


def test_add_fp16():

    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT16", [2])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    with tu.create_test_device() as device:
        session = popart.InferenceSession(
            fnModel=proto, dataFlow=dataFlow, deviceInfo=device
        )

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        inputs = {
            i1: np.array([1.0, 3.0], dtype=np.float16),
            i2: np.array([7.0, 8.0], dtype=np.float16),
        }
        stepio = popart.PyStepIO(inputs, anchors)

        session.run(stepio)

    assert np.allclose(anchors[o], np.array([8.0, 11.0], dtype=np.float16))


def test_add_variable_fp16():

    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT16", [2])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInitializedInputTensor(np.array([2.0, 4.0], dtype=np.float16))
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    with tu.create_test_device() as device:
        session = popart.InferenceSession(
            fnModel=proto, dataFlow=dataFlow, deviceInfo=device
        )

        session.prepareDevice()
        session.weightsFromHost()

        anchors = session.initAnchorArrays()

        inputs = {i1: np.array([1.0, 3.0], dtype=np.float16)}
        stepio = popart.PyStepIO(inputs, anchors)

        session.run(stepio)

    assert np.allclose(anchors[o], np.array([3.0, 7.0], dtype=np.float16))


def test_fp16transpose():
    """
    The compute graph :

     [c0] const f16 [5x6] ---> transpose to [6x5] -|
                                                   |--- add --> 6x5
     [d0] input f16 [1x6] ---> transpose to [6x1] -|

     This test compares popart to numpy, inference only

    """

    c0_data = np.random.randn(5, 6).astype(np.float16) * 1

    d0_data = np.random.randn(1, 6).astype(np.float16) * 1
    d0_info = popart.TensorInfo(d0_data)

    builder = popart.Builder()
    c0 = builder.aiOnnx.constant(c0_data, "c0")
    d0 = builder.addInputTensor(d0_info)

    c0_t = builder.aiOnnx.transpose([c0], [], "c0_transpose")
    d0_t = builder.aiOnnx.transpose([d0], [], "d0_transpose")

    o = builder.aiOnnx.add([d0_t, c0_t], "add")
    builder.addOutputTensor(o)

    proto = builder.getModelProto()
    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})
    with tu.create_test_device() as device:
        session = popart.InferenceSession(
            fnModel=proto, dataFlow=dataFlow, deviceInfo=device
        )

        session.prepareDevice()
        anchors = session.initAnchorArrays()
        inputs = {d0: d0_data}
        stepio = popart.PyStepIO(inputs, anchors)
        session.run(stepio)

    # numpy reference
    npout = np.transpose(c0_data) + np.transpose(d0_data)

    # lowering default tolerance for half-precision
    assert np.allclose(anchors[o], npout, atol=1e-4, rtol=1e-2)
