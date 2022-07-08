# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
# Test for basic importing

from test_session import PopartTestSession
import test_util as tu
import numpy as np


def test_patterns_default():
    import popart

    patterns = popart.Patterns()
    assert patterns.PreUniRepl is True
    assert patterns.PostNRepl is True
    assert patterns.SoftMaxGradDirect is True
    assert patterns.OpToIdentity is True
    assert patterns.SubtractArg1GradOp is True
    assert patterns.InPlace is True

    print(str(patterns))


def test_patterns_none():
    import popart

    patterns = popart.Patterns(popart.PatternsLevel.NoPatterns)
    assert patterns.PreUniRepl is False
    assert patterns.PostNRepl is False
    assert patterns.SoftMaxGradDirect is False
    assert patterns.OpToIdentity is False
    assert patterns.SubtractArg1GradOp is False
    assert patterns.InPlace is False

    print(str(patterns))


def test_patterns_all():
    import popart

    patterns = popart.Patterns(popart.PatternsLevel.All)
    assert patterns.PreUniRepl is True
    assert patterns.PostNRepl is True
    assert patterns.SoftMaxGradDirect is True
    assert patterns.OpToIdentity is True
    assert patterns.SubtractArg1GradOp is True
    assert patterns.InPlace is True

    print(str(patterns))


def test_patterns_modify():
    import popart

    patterns = popart.Patterns()
    patterns.PreUniRepl = False

    assert patterns.PreUniRepl is False
    assert patterns.PostNRepl is True
    assert patterns.SoftMaxGradDirect is True
    assert patterns.OpToIdentity is True
    assert patterns.SubtractArg1GradOp is True
    assert patterns.InPlace is True

    print(str(patterns))


def test_patterns_str():
    import popart

    patterns = popart.Patterns(["PostNRepl", "InPlace"])
    assert patterns.PostNRepl is True
    assert patterns.InPlace is True
    assert patterns.SoftMaxGradDirect is False


def test_loss_inputs_untouched():
    import popart

    height = 32
    batchesPerStep = 5
    samplesPerBatch = 48
    stepDataShape = [batchesPerStep, samplesPerBatch, height, height]

    input_data = np.zeros(stepDataShape).astype(np.float32)
    weights_data = np.zeros([height, height]).astype(np.float32)

    def init_builder(builder):
        i0 = builder.addInputTensor(input_data)
        w0 = builder.addInitializedInputTensor(weights_data)

        x = builder.aiOnnx.matmul([i0, w0])
        loss = builder.aiGraphcore.identityloss([x])
        builder.addOutputTensor(x)
        builder.setLoss(loss)

        return []

    session = PopartTestSession()
    session.mode = "train"
    session.patterns = popart.Patterns(popart.PatternsLevel.Default)
    with tu.create_test_device() as device:
        session.prepare(init_builder, device=device)
