# Test for basic importing

import pytest


def test_patterns_default():
    import poponnx

    patterns = poponnx.Patterns()
    assert (patterns.PreUniRepl == True)
    assert (patterns.PostNRepl == True)
    assert (patterns.SoftMaxGradDirect == True)
    assert (patterns.SplitConvBias == True)
    assert (patterns.OpToIdentity == True)
    assert (patterns.SubtractArg1GradOp == True)
    assert (patterns.InPlace0 == True)

    print(str(patterns))


def test_patterns_none():
    import poponnx

    patterns = poponnx.Patterns(poponnx.PatternsLevel.NONE)
    assert (patterns.PreUniRepl == False)
    assert (patterns.PostNRepl == False)
    assert (patterns.SoftMaxGradDirect == False)
    assert (patterns.SplitConvBias == False)
    assert (patterns.OpToIdentity == False)
    assert (patterns.SubtractArg1GradOp == False)
    assert (patterns.InPlace0 == False)

    print(str(patterns))


def test_patterns_all():
    import poponnx

    patterns = poponnx.Patterns(poponnx.PatternsLevel.ALL)
    assert (patterns.PreUniRepl == True)
    assert (patterns.PostNRepl == True)
    assert (patterns.SoftMaxGradDirect == True)
    assert (patterns.SplitConvBias == True)
    assert (patterns.OpToIdentity == True)
    assert (patterns.SubtractArg1GradOp == True)
    assert (patterns.InPlace0 == True)

    print(str(patterns))


def test_patterns_modify():
    import poponnx

    patterns = poponnx.Patterns()
    patterns.PreUniRepl = False

    assert (patterns.PreUniRepl == False)
    assert (patterns.PostNRepl == True)
    assert (patterns.SoftMaxGradDirect == True)
    assert (patterns.SplitConvBias == True)
    assert (patterns.OpToIdentity == True)
    assert (patterns.SubtractArg1GradOp == True)
    assert (patterns.InPlace0 == True)

    print(str(patterns))


def test_patterns_str():
    import poponnx

    patterns = poponnx.Patterns(["PostNRepl", "InPlace0"])
    assert (patterns.PostNRepl == True)
    assert (patterns.InPlace0 == True)
    assert (patterns.SoftMaxGradDirect == False)


def test_patterns_enum():
    import poponnx

    patterns = poponnx.Patterns(
        [poponnx.PatternType.POSTNREPL, poponnx.PatternType.INPLACE0])
    assert (patterns.PostNRepl == True)
    assert (patterns.InPlace0 == True)
    assert (patterns.SoftMaxGradDirect == False)
