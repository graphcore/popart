# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import popart


def test_enable_pattern():
    patterns = popart.Patterns()

    # a non-mandatory patterns
    pattern = "PadSum"
    enabled = patterns.isPatternEnabled(pattern)
    patterns.enablePattern(pattern, not enabled)
    assert (not enabled == patterns.isPatternEnabled(pattern))


def test_enable_inplace():
    patterns = popart.Patterns()

    # Turn off inplacing and verify it is off
    patterns.enablePattern("InPlace", False)
    assert not patterns.InPlace

    # Turn on inplacing and verify it is on
    patterns.enablePattern("InPlace", True)
    assert patterns.InPlace
