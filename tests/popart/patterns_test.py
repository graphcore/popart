# Test for basic importing

import pytest


def test_patterns_default():
    import popart

    patterns = popart.Patterns()
    assert (patterns.PreUniRepl == True)
    assert (patterns.PostNRepl == True)
    assert (patterns.SoftMaxGradDirect == True)
    assert (patterns.SplitConvBias == True)
    assert (patterns.OpToIdentity == True)
    assert (patterns.SubtractArg1GradOp == True)
    assert (patterns.InPlace == True)

    print(str(patterns))


def test_patterns_none():
    import popart

    patterns = popart.Patterns(popart.PatternsLevel.NONE)
    assert (patterns.PreUniRepl == False)
    assert (patterns.PostNRepl == False)
    assert (patterns.SoftMaxGradDirect == False)
    assert (patterns.SplitConvBias == False)
    assert (patterns.OpToIdentity == False)
    assert (patterns.SubtractArg1GradOp == False)
    assert (patterns.InPlace == False)

    print(str(patterns))


def test_patterns_all():
    import popart

    patterns = popart.Patterns(popart.PatternsLevel.ALL)
    assert (patterns.PreUniRepl == True)
    assert (patterns.PostNRepl == True)
    assert (patterns.SoftMaxGradDirect == True)
    assert (patterns.SplitConvBias == True)
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
    assert (patterns.SplitConvBias == True)
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

    patterns = popart.Patterns([popart.PreAliasPatternType.POSTNREPL])
    patterns.InPlace = True
    assert (patterns.PostNRepl == True)
    assert (patterns.InPlace == True)
    assert (patterns.SoftMaxGradDirect == False)
