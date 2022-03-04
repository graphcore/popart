# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import test_util as tu


def test_basic():
    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", [3])
    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)

    a1 = builder.aiOnnx.add([i1, i2])
    p1 = builder.aiGraphcore.nop([a1])
    a2 = builder.aiOnnx.add([i1, p1])
    p2 = builder.aiGraphcore.nop([a2])

    o = p2
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()
    opts.enableOutlining = False
    opts.enableOutliningCopyCostPruning = False

    with tu.create_test_device() as device:
        session = popart.InferenceSession(fnModel=proto,
                                          dataFlow=dataFlow,
                                          userOptions=opts,
                                          deviceInfo=device)

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        inputs = {
            i1: np.array([1., 2., 3.], dtype=np.float32),
            i2: np.array([4., 5., 6.], dtype=np.float32)
        }
        stepio = popart.PyStepIO(inputs, anchors)

        session.run(stepio)

    assert np.array_equal(anchors[o], np.array([6., 9., 12.],
                                               dtype=np.float32))


def test_builder_shape_inference():
    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", [3])
    i1 = builder.addInputTensor(shape)

    n1 = builder.aiGraphcore.nop([i1])
    m1 = builder.aiOnnx.mul([n1, i1])

    o = m1
    builder.addOutputTensor(o)

    nopShape = builder.getTensorShape(n1)
    print(f'nopShape: {nopShape}')
