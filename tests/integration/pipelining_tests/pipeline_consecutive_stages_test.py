# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import json

# 'import test_util' requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import test_util as tu


@tu.requires_ipu_model
def test_pipeline_stage_merging():
    np.random.seed(0)

    # With 3 stages the minimum pipeline cycles is 5
    # With 2 stages the minimum pipeline cycles is 3
    # So if the consecutive stages aren't fused an error will be thrown.
    gradient_accumulation = 3
    batch_size = 1
    hidden_size = 16

    input_shape = [batch_size, hidden_size]

    weight_data = np.random.normal(0, 0.02, [hidden_size, hidden_size]).astype(
        np.float32)

    builder = popart.Builder()

    x_in = builder.addInputTensor(popart.TensorInfo("FLOAT", input_shape),
                                  "x_in")

    weight_1 = builder.addInitializedInputTensor(weight_data, "weight_1")
    weight_2 = builder.addInitializedInputTensor(weight_data, "weight_2")
    weight_3 = builder.addInitializedInputTensor(weight_data, "weight_3")

    # Pipelining should combine stage 0 and 1.
    with builder.virtualGraph(0), builder.pipelineStage(0):
        x_0 = builder.aiOnnx.matmul([x_in, weight_1])

    with builder.virtualGraph(0), builder.pipelineStage(1):
        x_1 = builder.aiOnnx.matmul([x_0, weight_2])

    with builder.virtualGraph(1), builder.pipelineStage(2):
        o = builder.aiOnnx.matmul([x_1, weight_3])
        l1 = builder.aiGraphcore.l1loss([o], 0.1)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, [o])

    opts = popart.SessionOptions()
    opts.enableOutlining = False
    opts.enablePipelining = True
    opts.enableGradientAccumulation = True
    opts.accumulationFactor = gradient_accumulation
    opts.autoRecomputation = popart.RecomputationType.Pipeline
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    with tu.create_test_device(numIpus=2, opts={"compileIPUCode":
                                                False}) as device:
        session = popart.TrainingSession(fnModel=proto,
                                         dataFlow=dataFlow,
                                         userOptions=opts,
                                         loss=l1,
                                         optimizer=popart.ConstSGD(1e-9),
                                         deviceInfo=device)

        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))
        stashes = [op for op in ir["maingraph"] if op["type"] == "Stash"]
        stashedTensors = [stash["inputs"][0]["name"] for stash in stashes]

        assert {'x_in'} == set(stashedTensors)
