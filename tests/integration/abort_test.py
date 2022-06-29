# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import test_util as tu

import popart


def test_abort_unconditional():
    input_data = np.array(range(10), dtype=np.float32)
    builder = popart.Builder()
    t = builder.addInitializedInputTensor(input_data, "input")
    o = builder.aiOnnx.abs([t])

    builder.aiGraphcore.abort([])
    builder.addOutputTensor(o)
    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})
    opts = popart.SessionOptions()

    with tu.create_test_device() as device:
        session = popart.InferenceSession(fnModel=proto,
                                          dataFlow=dataFlow,
                                          userOptions=opts,
                                          deviceInfo=device)

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        inputs = {}
        stepio = popart.PyStepIO(inputs, anchors)
        with pytest.raises(popart.poplar_runtime_error) as e_info:
            session.run(stepio)
            assert (e_info.value.args[0].startswith("Abort Program"))


@tu.requires_ipu
def test_abort_conditional():
    input_data = np.array([0], dtype=np.float32)
    builder = popart.Builder()
    t = builder.addInitializedInputTensor(input_data, "input")
    o = builder.aiOnnx.abs([t])

    builder.aiGraphcore.abort([o])
    builder.addOutputTensor(o)
    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})
    opts = popart.SessionOptions()

    with tu.create_test_device() as device:
        session = popart.InferenceSession(fnModel=proto,
                                          dataFlow=dataFlow,
                                          userOptions=opts,
                                          deviceInfo=device)

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        inputs = {}
        stepio = popart.PyStepIO(inputs, anchors)

        session.run(stepio)


@tu.requires_ipu
def test_abort_conditional_exception():
    input_data = np.array([1], dtype=np.float32)
    builder = popart.Builder()
    t = builder.addInitializedInputTensor(input_data, "input")
    o = builder.aiOnnx.abs([t])

    builder.aiGraphcore.abort([o])
    builder.addOutputTensor(o)
    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})
    opts = popart.SessionOptions()

    with tu.create_test_device() as device:
        session = popart.InferenceSession(fnModel=proto,
                                          dataFlow=dataFlow,
                                          userOptions=opts,
                                          deviceInfo=device)

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        inputs = {}
        stepio = popart.PyStepIO(inputs, anchors)
        with pytest.raises(popart.poplar_runtime_error):
            session.run(stepio)
