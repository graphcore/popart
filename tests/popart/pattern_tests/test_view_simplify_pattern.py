import numpy as np
import popart
import json
import pytest

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def reshape(builder, x, outshape):
    return builder.reshape_const(builder.aiOnnx, [x], outshape)


def identity(builder, x, _):
    return builder.aiOnnx.identity([x])


@pytest.mark.parametrize(
    "a, b, target",
    [[reshape, reshape, "Reshape"], [identity, identity, "Identity"],
     [reshape, identity, "Reshape"], [identity, reshape, "Reshape"]])
def test_view_simplify(a, b, target):
    d1 = np.random.randn(10, 20).astype(np.float32)

    builder = popart.Builder()
    d = builder.addInputTensor("FLOAT", d1.shape)
    o = a(builder, d, [1, *d1.shape])
    o = b(builder, o, [*reversed(d1.shape)])

    opts = popart.SessionOptions()
    # ViewSimplifyPattern only runs when outlining
    opts.enableOutlining = True
    # Set the threshold high so nothing actually gets outlined.
    # This makes it easier to parse the IR.
    opts.outlineThreshold = 100000

    sess = popart.InferenceSession(fnModel=builder.getModelProto(),
                                   deviceInfo=tu.create_test_device(),
                                   dataFlow=popart.DataFlow(1, [o]))
    sess.prepareDevice()

    anchors = sess.initAnchorArrays()

    stepio = popart.PyStepIO({d: d1}, anchors)
    sess.weightsFromHost()

    sess.run(stepio)
    ir = json.loads(sess._serializeIr(popart.IrSerializationFormat.JSON))

    # Prune should have removed the first Op
    assert len(ir["maingraph"]) == 1
    # The remaining Op should be target.
    assert target in ir["maingraph"][0]["type"]
    assert np.allclose(anchors[o].flatten(), d1.flatten())
