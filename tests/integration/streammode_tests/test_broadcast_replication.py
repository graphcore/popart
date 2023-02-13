# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


@pytest.mark.parametrize("bps", [16])
@pytest.mark.parametrize("num_replicas", [2, 4])
@pytest.mark.parametrize("dim", [4, 8])
def test_broadcast_replication(bps, num_replicas, dim):
    builder = popart.Builder()
    ts_in = builder.addInputTensor(
        popart.TensorInfo("FLOAT", [dim, dim, dim]),
        settings=popart.InputSettings(popart.ReplicatedStreamMode.Broadcast),
        debugContext="input",
    )

    weights = np.random.rand(dim, dim).astype(np.float32)
    ts_w = builder.addInitializedInputTensor(weights, debugContext="weights")

    o = builder.aiOnnx.matmul([ts_in, ts_w])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()
    anchors = {o: popart.AnchorReturnType("ALL")}
    dataFlow = popart.DataFlow(bps, anchors)
    device = tu.create_test_device(numIpus=num_replicas)
    if tu.ipu_available(num_replicas):
        device = tu.create_test_device(numIpus=num_replicas)
    else:
        pytest.skip("No IPUS available for test options.")

    with device as device:
        opts = popart.SessionOptions()
        opts.enableReplicatedGraphs = True
        opts.replicatedGraphCount = num_replicas
        session = popart.InferenceSession(
            fnModel=proto, dataFlow=dataFlow, userOptions=opts, deviceInfo=device
        )

        session.prepareDevice()
        data_in = np.random.randn(bps, dim, dim, dim).astype(np.float32)

        anchors = session.initAnchorArrays()
        inputs = {ts_in: data_in}
        stepio = popart.PyStepIO(inputs, anchors)

        session.weightsFromHost()

        ref = np.matmul(data_in, weights).astype(np.float32)

        for i in range(num_replicas):
            session.run(stepio)
            out = anchors[o]
            out_i = np.squeeze(out[:, i])
            assert np.allclose(out_i, ref, equal_nan=False, atol=0.05)
