# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
from utils import contains_op_of_type


class TestNot:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(True, pir.bool)
            c = ops.logical_not(a)
        assert len(g.get_tensors()) == 2
        assert len(g.get_variables()) == 1
        assert contains_op_of_type("Not", _ir.op.NotOp, g)

    def test_needs_casting(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1, pir.int32)
            c = ops.logical_not(a)
        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 1
        assert contains_op_of_type("Not", _ir.op.NotOp, g)
