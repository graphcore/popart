# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
import numpy as np
from utils import contains_op_of_type
import pytest


@pytest.mark.parametrize("inplace", [True, False])
class TestDetach:
    def test_fn(self, inplace):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(np.ones((1, 2, 3)))
            if inplace:
                c = ops.detach_(a)
            else:
                c = ops.detach(a)
        assert len(g.get_tensors()) == 2
        assert len(g.get_variables()) == 1
        if inplace:
            assert contains_op_of_type("DetachInplace", _ir.op.DetachInplaceOp,
                                       g)
        else:
            assert contains_op_of_type("Detach", _ir.op.DetachOp, g)

    def test_dunder(self, inplace):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(np.ones((1, 2, 3)))
            if inplace:
                c = a.detach_()
            else:
                c = a.detach()
        assert len(g.get_tensors()) == 2
        assert len(g.get_variables()) == 1
        if inplace:
            assert contains_op_of_type("DetachInplace", _ir.op.DetachInplaceOp,
                                       g)
        else:
            assert contains_op_of_type("Detach", _ir.op.DetachOp, g)
