# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart.ir as pir
import popart.ir.ops as ops

import popart._internal.ir as _ir

from utils import contains_op_of_type


class TestNegate:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1)
            c = ops.negate(a)
        assert len(g.get_tensors()) == 2
        assert len(g.get_variables()) == 1
        assert contains_op_of_type("Neg", _ir.op.NegateOp, g)

    def test_dunder(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1)
            c = -a
        assert len(g.get_tensors()) == 2
        assert len(g.get_variables()) == 1
        assert contains_op_of_type("Neg", _ir.op.NegateOp, g)
