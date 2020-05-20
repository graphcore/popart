# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import test_util as tu


# test the case where the entire graph is computable as a const exprssion
def test_all_constexpr():
    d1 = np.random.rand(2, 2).astype(np.float32)
    d2 = np.random.rand(2, 2).astype(np.float32)
    d3 = np.random.rand(2, 2).astype(np.float32)
    d4 = np.random.rand(2, 2).astype(np.float32)

    datas = [np.random.rand(2, 2).astype(np.float32) for _ in range(4)]

    builder = popart.Builder()

    consts = [builder.aiOnnx.constant(data) for data in datas]
    a1 = builder.aiOnnx.add(consts[0:2])
    a2 = builder.aiOnnx.add(consts[2:4])
    a3 = builder.aiOnnx.add([a1, a2])

    out = a3
    builder.addOutputTensor(out)

    session = popart.InferenceSession(
        fnModel=builder.getModelProto(),
        dataFlow=popart.DataFlow(1, {out: popart.AnchorReturnType("All")}),
        patterns=popart.Patterns(popart.PatternsLevel.All),
        deviceInfo=tu.create_test_device())

    session.prepareDevice()
    session.weightsFromHost()

    anchors = session.initAnchorArrays()
    stepio = popart.PyStepIO({}, anchors)
    session.run(stepio)

    # check the result is correct
    assert (anchors[a3].shape == sum(datas).shape)
    assert (np.allclose(anchors[a3], sum(datas)))
