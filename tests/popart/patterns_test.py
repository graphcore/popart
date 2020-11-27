# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
# Test for basic importing

import pytest
from test_session import PopartTestSession
import numpy as np


def test_patterns_default():
    import popart

    patterns = popart.Patterns()
    assert (patterns.PreUniRepl == True)
    assert (patterns.PostNRepl == True)
    assert (patterns.SoftMaxGradDirect == True)
    assert (patterns.OpToIdentity == True)
    assert (patterns.SubtractArg1GradOp == True)
    assert (patterns.InPlace == True)

    print(str(patterns))


def test_patterns_none():
    import popart

    patterns = popart.Patterns(popart.PatternsLevel.NoPatterns)
    assert (patterns.PreUniRepl == False)
    assert (patterns.PostNRepl == False)
    assert (patterns.SoftMaxGradDirect == False)
    assert (patterns.OpToIdentity == False)
    assert (patterns.SubtractArg1GradOp == False)
    assert (patterns.InPlace == False)

    print(str(patterns))


def test_patterns_all():
    import popart

    patterns = popart.Patterns(popart.PatternsLevel.All)
    assert (patterns.PreUniRepl == True)
    assert (patterns.PostNRepl == True)
    assert (patterns.SoftMaxGradDirect == True)
    assert (patterns.OpToIdentity == True)
    assert (patterns.SubtractArg1GradOp == True)
    assert (patterns.InPlace == True)

    print(str(patterns))


def test_patterns_modify():
    import popart

    patterns = popart.Patterns()
    patterns.PreUniRepl = False

    assert (patterns.PreUniRepl == False)
    assert (patterns.PostNRepl == True)
    assert (patterns.SoftMaxGradDirect == True)
    assert (patterns.OpToIdentity == True)
    assert (patterns.SubtractArg1GradOp == True)
    assert (patterns.InPlace == True)

    print(str(patterns))


def test_patterns_str():
    import popart

    patterns = popart.Patterns(["PostNRepl", "InPlace"])
    assert (patterns.PostNRepl == True)
    assert (patterns.InPlace == True)
    assert (patterns.SoftMaxGradDirect == False)


def test_patterns_enum():
    import popart

    patterns = popart.Patterns([popart.PreAliasPatternType.PostNRepl])
    patterns.InPlace = True
    assert (patterns.PostNRepl == True)
    assert (patterns.InPlace == True)
    assert (patterns.SoftMaxGradDirect == False)


def test_loss_inputs_untouched():
    import popart

    height = 32
    batchesPerStep = 5
    samplesPerBatch = 48
    samplesPerMicroBatch = samplesPerBatch
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
    session.mode = 'train'
    session.patterns = popart.Patterns(popart.PatternsLevel.Default)
    session.prepare(init_builder)
