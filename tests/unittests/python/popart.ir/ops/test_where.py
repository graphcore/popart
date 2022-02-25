# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestWhere:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            b = pir.variable([True, True, False])
            lhs = pir.variable(np.arange(3))
            rhs = pir.variable(np.arange(3)[::-1])
            c = ops.where(b, lhs, rhs)

        assert c.shape == lhs.shape
        assert len(g.tensors) == 4
        assert contains_op_of_type("Where", _ir.op.WhereOp, g)

    def test_cast(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            b = pir.variable([1, 1, 0], dtype=pir.int32)
            lhs = pir.variable(np.arange(3))
            rhs = pir.variable(np.arange(3)[::-1])
            c = ops.where(b, lhs, rhs)

        assert c.shape == lhs.shape
        assert len(g.tensors) == 5
        assert contains_op_of_type("Where", _ir.op.WhereOp, g)

    def test_dunder(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            x = pir.variable(np.arange(3))
            mask = pir.variable([True, False, True])
            c = x[mask]

        assert c.shape == x.shape
        assert len(g.tensors) == 4
        assert contains_op_of_type("Where", _ir.op.WhereOp, g)
