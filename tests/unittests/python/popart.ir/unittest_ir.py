# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import popart._internal.ir as _ir
"""
This file should contain unittests to check the correct working of the
popart.ir package. This is a public-facing API. Tests in this file should be
quick to run (these are explicitly not integration tests).
"""


# NOTE: This is a placeholder until we have better tests.
def test_ir_construction():
    """ Test that we can construct an popart.ir.Ir object. """
    ir = _ir.Ir()
