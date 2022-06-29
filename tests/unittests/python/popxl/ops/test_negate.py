# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops

import popart._internal.ir as _ir

from utils import contains_op_of_type


class TestNegate:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            _ = ops.negate(a)
        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        assert contains_op_of_type("Neg", _ir.op.NegateOp, g)

    def test_dunder(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            _ = -a
        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        assert contains_op_of_type("Neg", _ir.op.NegateOp, g)
