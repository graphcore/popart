# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import popxl
import popxl.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestWhere:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            b = popxl.variable([True, True, False])
            lhs = popxl.variable(np.arange(3))
            rhs = popxl.variable(np.arange(3)[::-1])
            c = ops.where(b, lhs, rhs)

        assert c.shape == lhs.shape
        assert len(g.tensors) == 4
        assert contains_op_of_type("Where", _ir.op.WhereOp, g)

    def test_cast(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            b = popxl.variable([1, 1, 0], dtype=popxl.int32)
            lhs = popxl.variable(np.arange(3))
            rhs = popxl.variable(np.arange(3)[::-1])
            c = ops.where(b, lhs, rhs)

        assert c.shape == lhs.shape
        assert len(g.tensors) == 5
        assert contains_op_of_type("Where", _ir.op.WhereOp, g)

    def test_dunder(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            x = popxl.variable(np.arange(3))
            mask = popxl.variable([True, False, True])
            c = x[mask]

        assert c.shape == x.shape
        assert len(g.tensors) == 4
        assert contains_op_of_type("Where", _ir.op.WhereOp, g)
