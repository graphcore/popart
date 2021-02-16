# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import torch
import torch.nn as nn

import popart
from op_tester import op_tester

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


@tu.requires_ipu_model
def test_call_pipelined(op_tester):
    d0 = np.asarray([2, -1]).astype(np.int32)
    d1 = np.asarray([-4, 3]).astype(np.int32)

    builder = popart.Builder()

    i0 = builder.addInputTensor(popart.TensorInfo("INT32", [2]))
    i1 = builder.addInputTensor(popart.TensorInfo("INT32", [2]))

    subgraph_builder = builder.createSubgraphBuilder()

    info = popart.TensorInfo("INT32", [2])
    sgi0 = subgraph_builder.addInputTensor(info)
    sgi1 = subgraph_builder.addInputTensor(info)

    subgraph_builder.addOutputTensor(subgraph_builder.aiOnnx.add([sgi0, sgi1]))

    with builder.virtualGraph(0), builder.pipelineStage(0):
        act = builder.aiGraphcore.call([i0, i1], 1, subgraph_builder)[0]
    with builder.virtualGraph(1), builder.pipelineStage(1):
        out = builder.aiGraphcore.call([act, i1], 1, subgraph_builder)[0]

    builder.addOutputTensor(out)

    opts = popart.SessionOptions()
    opts.enablePipelining = True
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                      dataFlow=popart.DataFlow(10, [out]),
                                      userOptions=opts,
                                      deviceInfo=tu.create_test_device(
                                          numIpus=4, tilesPerIPU=20))
