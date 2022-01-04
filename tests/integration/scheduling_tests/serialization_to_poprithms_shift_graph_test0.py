# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import popart
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def test_add_serialization(tmpdir):
    """
    enable serialization to poprithms shift graphs, and assert that 
    at least 1 json file is written to a temporary directory.
    """

    builder = popart.Builder()
    shape = popart.TensorInfo("FLOAT", [2])
    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)
    proto = builder.getModelProto()
    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()
    opts.serializedPoprithmsShiftGraphsDir = str(tmpdir)

    session = popart.InferenceSession(fnModel=proto,
                                      dataFlow=dataFlow,
                                      userOptions=opts,
                                      deviceInfo=tu.create_test_device())

    session.prepareDevice()

    jsons = [os.path.join(tmpdir, x) for x in os.listdir(tmpdir)]
    assert (len(jsons) > 0)
