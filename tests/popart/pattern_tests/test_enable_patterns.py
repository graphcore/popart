# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import popart


def test_enable_pattern():
    patterns = popart.Patterns()
    pattern = str(patterns).split(" ")[0]
    enabled = patterns.isPatternEnabled(pattern)
    patterns.enablePattern(pattern, not enabled)
    assert (not enabled == patterns.isPatternEnabled(pattern))
