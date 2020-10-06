# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import popart
import numpy as np
import pytest


# Regression test for T27637
@pytest.mark.parametrize("ipu", {True, False})
@pytest.mark.parametrize("devices", {1, 2})
def test_stepio_shuffle_test(ipu, devices):
    builder = popart.Builder()
    i = builder.addInputTensor("FLOAT", [2, 2])
    o = builder.aiOnnx.sqrt([i])
    fnModel = builder.getModelProto()

    def checkDeterministicInference(fnModel, bps, deviceInfo):
        print("Running model ", fnModel, " bps: ", bps, "device: ", deviceInfo)
        data = np.random.rand(bps, 2, 2).astype(np.float32)

        # Load trained weights directly from onnx
        deviceConfig = {'numIPUs': 1}
        session = popart.InferenceSession(fnModel=fnModel,
                                          dataFlow=popart.DataFlow(bps, [o]),
                                          deviceInfo=deviceInfo)

        session.prepareDevice()
        session.weightsFromHost()
        anchors = session.initAnchorArrays()
        stepio = popart.PyStepIO({i: data}, anchors)
        session.run(stepio)
        ref_out = np.copy(anchors[o])

        # Verify deterministic runs for same weights
        session.run(stepio)
        assert np.allclose(ref_out, anchors[o])

    dev = None
    if ipu:
        dev = popart.DeviceManager().acquireAvailableDevice(1)
    else:
        dev = popart.DeviceManager().createCpuDevice()

    checkDeterministicInference(fnModel, devices, dev)  # pass
