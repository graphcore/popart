# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import json
import numpy as np
import pytest

import popart

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


# This test is currently expected to fail, until the cause of T23064 is fixed.
@pytest.mark.xfail(raises=AssertionError)
@tu.requires_ipu_model
@pytest.mark.parametrize("constantWeights", [True, False])
def test_inplace_weight_add(constantWeights):
    r"""
    Test an inplacing add on the weight. I.e:
    input
      |
      sin   weight
        \   /
         add
          |
         cos
          |

    Which becomes:
                 input
                   |
        weight += sin
          |
         cos
          |

    The sin and cos are just ensuring the network doesn't get optimised
    to nothing.

    Requirements for reproducing:
    1. Inference only
    2. Force inplacing on the weight, not the input

    constantWeights seems not to affect the result?

    Then inplacing on will not match inplacing off, which is incorrect.
    """

    input_size = 500
    np.random.seed(0)
    data = np.random.rand(input_size).astype(np.float32)
    weight = np.random.rand(input_size).astype(np.float32)

    def create_session(inplacing):
        builder = popart.Builder()

        input_ = builder.addInputTensor(popart.TensorInfo("FLOAT", data.shape),
                                        "data")
        sin = builder.aiOnnx.sin([input_], "sin")
        w = builder.addInitializedInputTensor(weight, 'input_weights')
        add = builder.aiOnnx.add([sin, w], "add")
        builder.setInplacePreferences(
            add, {"AddRhsInplace": +1e8})  # force weight inplacing
        cos = builder.aiOnnx.cos([add], "cos")

        loss = builder.aiGraphcore.l1loss([cos],
                                          0.1,
                                          reduction=popart.ReductionType.Mean)

        builder.addOutputTensor(loss)

        patterns = popart.Patterns(popart.PatternsLevel.Default)
        patterns.InPlace = inplacing

        opts = popart.SessionOptions()
        opts.constantWeights = constantWeights

        # Training seems to force inplacing off the weight tensor and on
        # to the activation.
        session = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            dataFlow=popart.DataFlow(1, [loss]),
            deviceInfo=tu.create_test_device(),
            userOptions=opts,
            patterns=patterns,
        )

        session.prepareDevice()

        anchorRets = session.initAnchorArrays()

        inputs = {"data": data.copy()}
        stepio = popart.PyStepIO(inputs, anchorRets)

        session.weightsFromHost()
        return session, stepio, anchorRets

    session1, stepio1, anchors1 = create_session(True)
    session2, stepio2, anchors2 = create_session(False)

    def verify_inplacing(session, count):
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        inplaceAdds = [
            op for op in ir['maingraph'] if op['type'] == 'AddRhsInplace'
        ]
        assert (len(inplaceAdds) == count)

    verify_inplacing(session1, 1)
    verify_inplacing(session2, 0)

    for i in range(5):
        session1.run(stepio1)
        session2.run(stepio2)
        for key in anchors1:
            print("Step", i, "Testing:", key)
            assert np.allclose(anchors1[key], anchors2[key])
