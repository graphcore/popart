# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir


def test_ir_construction():
    """ Test that we can construct a popart._internal.ir.Ir object. """
    _ir.Ir()
