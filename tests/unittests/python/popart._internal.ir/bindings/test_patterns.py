# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
from utils import *

import popart
import popart._internal.ir as _ir


@pytest.mark.parametrize("patterns_level", [
    _ir.patterns.PatternsLevel.NoPatterns, _ir.patterns.PatternsLevel.Minimal,
    _ir.patterns.PatternsLevel.Default, _ir.patterns.PatternsLevel.All
])
def test_check_applying_patterns_level(
        patterns_level: _ir.patterns.PatternsLevel) -> None:
    """Test you can set the patterns object via the PatternsLevelEnum

    Args:
        patterns_level (_ir.patterns.PatternsLevel): The patterns level enum.
    """
    ir = _ir.Ir()
    g1Id = _ir.GraphId("g1")
    _ = ir.createGraph(g1Id)
    g2Id = _ir.GraphId("g2")
    _ = ir.createGraph(g2Id)

    for g in ir.getAllGraphs():
        in0 = add_actgrad_tensor("in0", [1, 2, 3], g)
        in1 = add_random_tensor("in1", _ir.TensorType.Variable, [1, 2, 3], g)
        out0 = add_actgrad_tensor("out0", [1, 2, 3], g)
        ins = {0: in0, 1: in1}
        outs = {0: out0}
        _ = create_new_op(ins, outs, "AddOp", g)

    p = _ir.patterns.Patterns(patterns_level)

    if patterns_level == _ir.patterns.PatternsLevel.NoPatterns:
        p = p.enableRuntimeAsserts(False)

    ir.setPatterns(p)

    for g in ir.getAllGraphs():
        ir.applyPreAliasPatterns(g)
        ir.applyInplacePattern(g)


PATTERN_NAMES = _ir.patterns.Patterns.getAllPreAliasPatternNames()


@pytest.mark.parametrize("pattern", PATTERN_NAMES)
def test_get_set_patterns(pattern: str) -> None:
    """Check you can get/set patterns by name.

    Args:
        pattern (str): The Pattern name.
    """
    p = _ir.patterns.Patterns(_ir.patterns.PatternsLevel.NoPatterns)

    assert not p.isPatternEnabled(pattern)
    p.enablePattern(pattern, True)
    assert p.isPatternEnabled(pattern)


@pytest.mark.parametrize("pattern", PATTERN_NAMES)
def test_get_set_patterns_constructor(pattern: str) -> None:
    """Check you can use the constructor and pattern name to create patterns.

    Args:
        pattern (str): The Pattern name.
    """
    p = _ir.patterns.Patterns([pattern])

    assert p.isPatternEnabled(pattern)
    if not _ir.patterns.Patterns.isMandatory(pattern):
        p.enablePattern(pattern, False)
        assert not p.isPatternEnabled(pattern)
    else:
        with pytest.raises(popart.popart_exception) as e_info:
            p.enablePattern(pattern, False)
            assert e_info.value.args[
                0] == f"Pattern '{pattern}' must be enabled"
