# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops
from utils import contains_op_of_type


class TestNot:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(True, popxl.bool)
            c = ops.logical_not(a)
        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        assert contains_op_of_type("Not", _ir.op.NotOp, g)

    def test_needs_casting(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1, popxl.int32)
            c = ops.logical_not(a)
        assert len(g.tensors) == 3
        assert len(g.variables) == 1
        assert contains_op_of_type("Not", _ir.op.NotOp, g)

    def test_dunder(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(True, popxl.bool)
            c = ~a
        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        assert contains_op_of_type("Not", _ir.op.NotOp, g)
