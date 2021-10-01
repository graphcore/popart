# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
from utils import contains_op_of_type


class TestAnd:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(True, pir.bool)
            b = pir.variable(False, pir.bool)
            c = ops.logical_and(a, b)
        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 2
        assert contains_op_of_type("And", _ir.op.AndOp, g)

    def test_needs_casting(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1, pir.int32)
            b = pir.variable(0, pir.int32)
            c = ops.logical_and(a, b)
        assert len(g.get_tensors()) == 5
        assert len(g.get_variables()) == 2
        assert contains_op_of_type("And", _ir.op.AndOp, g)