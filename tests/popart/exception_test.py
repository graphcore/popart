# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


# test that we can get the graph & summary report after an out of memory exception
# This test currently requires hardware, as the ipu model does not throw an exception
# when it run's out of memory
@tu.requires_ipu
def test_out_of_memory_exception():
    deviceInfo = tu.create_test_device(1, tilesPerIPU=tu.USE_ALL_TILES)

    builder = popart.Builder()

    inputs = [
        builder.addInputTensor(popart.TensorInfo("FLOAT", [2000, 2000]))
        for i in range(8)
    ]

    # Matmul every input against every other input
    activations = []
    for a in inputs:
        for b in inputs:
            c = builder.aiOnnx.matmul([a, b])
            activations.append(c)

    # Sum all the activations together
    out = builder.aiOnnx.sum(activations)

    builder.addOutputTensor(out)

    options = popart.SessionOptions()
    options.defaultPrefetchBufferingDepth = 1
    options.engineOptions = {"debug.allowOutOfMemory": "true"}
    patterns = popart.Patterns(popart.PatternsLevel.NoPatterns)
    patterns.enableRuntimeAsserts(False)

    session = popart.InferenceSession(
        fnModel=builder.getModelProto(),
        dataFlow=popart.DataFlow(1, {out: popart.AnchorReturnType("All")}),
        userOptions=options,
        patterns=patterns,
        deviceInfo=deviceInfo)

    with pytest.raises(popart.popart_exception) as e:
        session.prepareDevice()

    assert e.type == popart.session.OutOfMemoryException
    print(e.value.getSummaryReport())
    print(e.value.getGraphReport())
    assert e.value.args[0].startswith(
        "Out of memory: Cannot fit all variable data onto one or more tiles")

    session.getTensorTileMap()
