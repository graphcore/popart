# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import popart
import pytest
import tempfile

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
    deviceInfo = tu.create_test_device(1, tilesPerIPU=4)

    builder = popart.Builder()

    inputs = [
        builder.addInputTensor(popart.TensorInfo("FLOAT", [100, 100]))
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
    tempDir = tempfile.TemporaryDirectory()
    options.engineOptions = {
        "debug.allowOutOfMemory": "true",
        "autoReport.outputGraphProfile": "true",
        "autoReport.directory": tempDir.name
    }
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
    assert len(e.value.getProfilePath()) > 0
    assert e.value.args[0].startswith(
        "Out of memory: Cannot fit all variable data onto one or more tiles")
