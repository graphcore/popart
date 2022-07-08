# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops
import numpy as np
from utils import contains_op_of_type
import pytest


@pytest.mark.parametrize("inplace", [True, False])
class TestDetach:
    def test_fn(self, inplace):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones((1, 2, 3)))
            if inplace:
                _ = ops.detach_(a)
            else:
                _ = ops.detach(a)
        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        if inplace:
            assert contains_op_of_type("DetachInplace", _ir.op.DetachInplaceOp, g)
        else:
            assert contains_op_of_type("Detach", _ir.op.DetachOp, g)

    def test_dunder(self, inplace):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones((1, 2, 3)))
            if inplace:
                _ = a.detach_()
            else:
                _ = a.detach()
        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        if inplace:
            assert contains_op_of_type("DetachInplace", _ir.op.DetachInplaceOp, g)
        else:
            assert contains_op_of_type("Detach", _ir.op.DetachOp, g)
