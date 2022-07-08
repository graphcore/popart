# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir


def test_source_location_construction():
    """Test that we can construct a popart._internal.ir.SourceLocation object."""
    _ir.SourceLocation("foo", "bar.py", 10)


def test_debug_context_construction():
    """Test that we can construct a popart._internal.ir.DebugContext object."""
    sl = _ir.SourceLocation("foo", "bar.py", 10)
    di = _ir.DebugInfo(_ir.DebugContext(), "test")
    _ir.DebugContext("baz", sl)
    _ir.DebugContext()
    _ir.DebugContext("baz")
    _ir.DebugContext(di, "baz")


def test_debug_context_implicit_conversion():
    """Test that we can implicitly convert a popart._internal.ir.DebugContext
    object to string.
    """
    # The first argument to DebugInfo is a DebugContext, but a string is given.
    _ir.DebugInfo("dc", "test")
