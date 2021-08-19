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

    input_size = 31
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
            op for op in ir['maingraph'] if op['type'] == 'AddLhsInplace'
        ]
        assert (len(inplaceAdds) == count)

    verify_inplacing(session1, 1)

    for i in range(5):
        session1.run(stepio1)
        session2.run(stepio2)
        for key in anchors1:
            print("Step", i, "Testing:", key)
            assert np.allclose(anchors1[key], anchors2[key])


@pytest.mark.parametrize("constantWeights", [True, False])
def test_non_modifying_inplace(constantWeights):
    r"""
    Check that the add is still inplaced on the left, despite the inplace transpose.

    input  w
     |     |
     |   transpose < highest in place priority
      \   /
       add < lower in place (RHS) priority
        |
    """
    input_size = [5, 7, 3]
    np.random.seed(0)
    data = np.random.rand(*input_size).astype(np.float32)
    weight = np.random.rand(7, 3, 5).astype(np.float32)

    def create_session(inplacing):
        builder = popart.Builder()

        input_ = builder.addInputTensor(popart.TensorInfo("FLOAT", data.shape),
                                        "data")
        w = builder.addInitializedInputTensor(weight, 'input_weights')

        # If we set the priorities to the same, it will choose to inplace the
        # Rhs over the transpose. But we want it to try to Rhs the inplace
        trans = builder.aiOnnx.transpose([w], [2, 0, 1], "transpose")
        builder.setInplacePreferences(trans, {"TransposeInplace": +1e8})

        add = builder.aiOnnx.add([input_, trans], "add")
        builder.setInplacePreferences(add, {"AddRhsInplace": +1e7})

        loss = builder.aiGraphcore.l1loss([add],
                                          0.1,
                                          reduction=popart.ReductionType.Mean)

        builder.addOutputTensor(loss)

        patterns = popart.Patterns(popart.PatternsLevel.Default)
        patterns.InPlace = inplacing

        opts = popart.SessionOptions()
        opts.constantWeights = constantWeights

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

    def verify_inplacing(session, count, type_):
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))
        inplaces = [op for op in ir['maingraph'] if op['type'] == type_]

        assert len(inplaces) == count

    if constantWeights:
        # Transpose gets optimised out with constant weights.
        verify_inplacing(session1, 1, "AddLhsInplace")
    else:
        verify_inplacing(session1, 1, "TransposeInplace")
        verify_inplacing(session1, 1, "AddLhsInplace")

    for i in range(5):
        session1.run(stepio1)
        session2.run(stepio2)
        for key in anchors1:
            print("Step", i, "Testing:", key)
            assert np.allclose(anchors1[key], anchors2[key])


@pytest.mark.parametrize("constantWeights", [True, False])
def test_non_modifying_inplace_2(constantWeights):
    r"""
    Only way we can observe this issue is in training,
    otherwise the graph gets optimized to nothing.

    input  w
     |     |
     |   transpose < highest inplace priority
     |     |
     |   reshape < low inplace priority
      \   /
       add < higher inplace priority
        |
       l1

    This should become:

    input  w
     |     |
     |   transposeInplace
     |     |
     |   reshape < NOT inplace
      \   /
    addRHSinplace
        |
       l1
    """
    input_size = [2, 4, 3]
    np.random.seed(0)
    data = np.random.rand(*input_size).astype(np.float32)
    weight = np.random.rand(2, 3, 2, 2).astype(np.float32)

    def create_session(inplacing):
        builder = popart.Builder()

        input_ = builder.addInputTensor(popart.TensorInfo("FLOAT", data.shape),
                                        "data")
        w = builder.addInitializedInputTensor(weight, 'input_weights')

        shape = builder.aiOnnx.constant(np.array(input_size))

        transpose = builder.aiOnnx.transpose([w], [0, 2, 1, 3], "transpose")
        builder.setInplacePreferences(transpose, {"TransposeInplace": +1e8})
        reshape = builder.aiOnnx.reshape([transpose, shape], "reshape")
        builder.setInplacePreferences(reshape, {"ReshapeInplace": +1e6})

        add = builder.aiOnnx.add([input_, reshape], "add")
        builder.setInplacePreferences(add, {"AddRhsInplace": +1e7})

        loss = builder.aiGraphcore.l1loss([add],
                                          0.1,
                                          reduction=popart.ReductionType.Mean)

        builder.addOutputTensor(loss)

        patterns = popart.Patterns(popart.PatternsLevel.Default)
        patterns.InPlace = inplacing

        opts = popart.SessionOptions()
        opts.constantWeights = constantWeights

        session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                         dataFlow=popart.DataFlow(1, [loss]),
                                         deviceInfo=tu.create_test_device(),
                                         userOptions=opts,
                                         patterns=patterns,
                                         loss=loss,
                                         optimizer=popart.ConstSGD(1e-3))

        session.prepareDevice()

        anchorRets = session.initAnchorArrays()

        inputs = {"data": data.copy()}
        stepio = popart.PyStepIO(inputs, anchorRets)

        session.weightsFromHost()
        return session, stepio, anchorRets

    session1, stepio1, anchors1 = create_session(True)
    session2, stepio2, anchors2 = create_session(False)

    def verify_inplacing(session, count, type_):
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        # Hacky way to remove backwards ops.
        ir = [
            op for op in ir['maingraph']
            if not op['outputs'][0]['name'].startswith("Gradient___")
        ]

        inplaces = [op for op in ir if op['type'] == type_]
        assert len(inplaces) == count

    verify_inplacing(session1, 1, "TransposeInplace")
    verify_inplacing(session1, 0, "ReshapeInplace")
    verify_inplacing(session1, 1, "Reshape")
    verify_inplacing(session1, 1, "AddRhsInplace")

    for i in range(5):
        session1.run(stepio1)
        session2.run(stepio2)
        for key in anchors1:
            print("Step", i, "Testing:", key)
            assert np.allclose(anchors1[key], anchors2[key])
