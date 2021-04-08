import numpy as np
import popart
import json
import pytest

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def test_collapse_reshape():
    d1 = np.random.randn(10, 20).astype(np.float32)

    builder = popart.Builder()
    d = builder.addInputTensor("FLOAT", d1.shape)
    o = builder.reshape_const(builder.aiOnnx, [d], [1, *d1.shape])
    o = builder.reshape_const(builder.aiOnnx, [o], [*reversed(d1.shape)])

    sess = popart.InferenceSession(fnModel=builder.getModelProto(),
                                   deviceInfo=tu.create_test_device(),
                                   dataFlow=popart.DataFlow(1, [o]))
    sess.prepareDevice()

    anchors = sess.initAnchorArrays()

    stepio = popart.PyStepIO({d: d1}, anchors)
    sess.weightsFromHost()

    sess.run(stepio)
    ir = json.loads(sess._serializeIr(popart.IrSerializationFormat.JSON))
    assert len(
        list(filter(lambda op: "Reshape" in op["type"], ir["maingraph"]))) == 1
    assert np.allclose(anchors[o].flatten(), d1.flatten())
