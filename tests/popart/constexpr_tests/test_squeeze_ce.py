import numpy as np
import popart
import json
import pprint

# importing test_session requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from test_session import TestSession


def test_basic_squeeze():
    np.random.seed(1)

    input_data = np.random.rand(4, 3).astype(np.float32)
    const_data = np.random.rand(1, 4, 1, 3, 1).astype(np.float32)

    squeeze_id = ""

    def init_builder(builder):
        nonlocal squeeze_id
        d0 = builder.addInputTensor(input_data, 'data0')
        c0 = builder.aiOnnx.constant(const_data)

        x = builder.aiOnnx.squeeze([c0], axes=[0, 2, 4])
        # adding identity to allow squeeze to be const folded
        x = builder.aiOnnx.identity([x])
        squeeze_id = x

        x = builder.aiOnnx.add([d0, x])

        builder.addOutputTensor(x)

        loss = builder.addL1Loss(x, 'l1LossVal', 0.1, popart.ReductionType.Sum)
        return [x, squeeze_id]

    def ref():
        axes = [4, 2, 0]
        x = const_data
        for axis in axes:
            x = np.squeeze(x, axis)
        return x

    session = TestSession()

    # test a pipeline stage appearing on multiple virtual graphs
    session.prepare(init_builder)

    anchors = session.run()

    # Check the squeeze op was removed
    ir = json.loads(
        session._session._serializeIr(popart.IrSerializationFormat.JSON))
    pprint.pprint(ir)
    ops = ir['maingraph']
    # There should only be 2 ops and none of them should be squeeze
    assert len(ops) == 2
    assert 'Squeeze' not in [i['type'] for i in ops]

    assert np.array_equal(anchors[squeeze_id], ref())
