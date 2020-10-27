# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import pprint
import json
import platform

# 'import test_util' requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import test_util as tu


@tu.requires_ipu_model
def test_full_recompute_pipelining(tmpdir):
    np.random.seed(0)

    gradient_accumulation = 5
    batch_size = 1
    hidden_size = 16

    input_shape = [batch_size, hidden_size]

    weight_data = np.random.normal(0, 0.02, [hidden_size, hidden_size]).astype(
        np.float32)

    input_data = np.random.normal(
        0, 0.02, [gradient_accumulation] + input_shape).astype(np.float32)

    def run_test(mode=None, verify=None):
        builder = popart.Builder()

        def norm(input_x):
            gamma = builder.addInitializedInputTensor(
                np.ones(hidden_size, np.float32), "Gamma")
            beta = builder.addInitializedInputTensor(
                np.zeros(hidden_size, np.float32), "Beta")
            return builder.aiGraphcore.groupnormalization(
                [input_x, gamma, beta], 1)[0]

        x_in = builder.addInputTensor(popart.TensorInfo("FLOAT", input_shape),
                                      "x_in")

        weight_1 = builder.addInitializedInputTensor(weight_data, "weight_1")
        weight_2 = builder.addInitializedInputTensor(weight_data, "weight_2")
        weight_3 = builder.addInitializedInputTensor(weight_data, "weight_3")

        with builder.virtualGraph(0), builder.pipelineStage(0):
            x_0 = builder.aiOnnx.matmul([x_in, weight_1])
            x_0 = norm(x_0)

            # If recomputeOutputs was used directly on `x_0` all 3 outputs
            # of groupnormalization would be stashed.
            # By using a checkpointOutput only 1 output will be stashed and the
            # rest will be recomputed.
            x_0 = builder.checkpointOutput([x_0])[0]

            x_1 = builder.aiOnnx.matmul([x_0, weight_2])
            x_1 = norm(x_1)
            x_1 = builder.aiOnnx.add([x_0, x_1])

        with builder.virtualGraph(1), builder.pipelineStage(1):
            o = builder.aiOnnx.matmul([x_1, weight_3])
            l1 = builder.aiGraphcore.l1loss([o], 0.1)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(1, [
            o,
            popart.reservedGradientPrefix() + weight_1,
            popart.reservedGradientPrefix() + weight_2,
            popart.reservedGradientPrefix() + weight_3,
        ])

        opts = popart.SessionOptions()
        opts.enableOutlining = False
        opts.enablePipelining = True
        opts.enableGradientAccumulation = True
        opts.accumulationFactor = gradient_accumulation
        opts.optimizerStateTensorLocationSettings.location.storage = popart.TensorStorage.OffChip
        if mode is not None:
            opts.autoRecomputation = mode
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual

        session = popart.TrainingSession(fnModel=proto,
                                         dataFlow=dataFlow,
                                         userOptions=opts,
                                         loss=l1,
                                         optimizer=popart.Adam({}),
                                         deviceInfo=tu.create_test_device(
                                             numIpus=2,
                                             opts={"compileIPUCode": False}))

        session.prepareDevice()

        session.weightsFromHost()

        anchors = session.initAnchorArrays()

        inputs = {x_in: input_data}
        stepio = popart.PyStepIO(inputs, anchors)

        for _ in range(10):
            session.run(stepio)

        if verify is not None:
            verify(session, x_0)

        return anchors

    def verify(session, mid_stash):
        ''' Verify the the matmul in the main graphs is correct'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))
        stashes = [op for op in ir["maingraph"] if op["type"] == "Stash"]
        stashedTensors = [stash["inputs"][0]["name"] for stash in stashes]

        assert {'x_in', mid_stash} == set(stashedTensors)

    n_anchors = run_test()
    p_anchors = run_test(popart.RecomputationType.Pipeline, verify)

    for key in n_anchors:
        assert np.allclose(n_anchors[key], p_anchors[key])
