# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import numpy as np

# 'import test_util' requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import test_util as tu


@tu.requires_ipu_model
def test_explicit_pipelining_0():
    """
    Numerical equivalence test: [Implicit pipelinng + Implicit host i/o] vs
                                [Explicit pipelinng + Explicit host i/o]
    """

    def getAnchors(isExplicit):
        bps = 3
        np.random.seed(1)
        builder = popart.Builder()

        t0 = builder.addInputTensor("FLOAT", [2, 2])
        t1_data = np.random.rand(2, 2).astype(np.float32)
        t1 = builder.addInitializedInputTensor(t1_data)

        with builder.virtualGraph(0), builder.pipelineStage(0):
            t2 = builder.aiOnnx.matmul([t0, t1])

        with builder.virtualGraph(1), builder.pipelineStage(1):
            t3 = builder.aiOnnx.relu([t2])

        with builder.virtualGraph(2), builder.pipelineStage(2):
            t4 = builder.aiOnnx.softmax([t3])

        opts = popart.SessionOptions()
        opts.enablePipelining = True
        opts.explicitRecomputation = True
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual

        opts.enableExplicitIR(isExplicit)

        with tu.create_test_device(numIpus=3) as device:
            session = popart.InferenceSession(
                fnModel=builder.getModelProto(),
                dataFlow=popart.DataFlow(bps, [t4]),
                deviceInfo=device,
                userOptions=opts,
            )

            session.prepareDevice()

            anchors = session.initAnchorArrays()

            t0_data = np.random.rand(bps, 2, 2).astype(np.float32)
            stepio = popart.PyStepIO({t0: t0_data}, anchors)
            session.run(stepio)

        return anchors

    explicitAnchors = getAnchors(isExplicit=True)
    implicitAnchors = getAnchors(isExplicit=False)

    for key in explicitAnchors:
        assert np.allclose(explicitAnchors[key], implicitAnchors[key])
