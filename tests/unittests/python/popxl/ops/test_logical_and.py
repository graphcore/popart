# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops
from utils import contains_op_of_type


class TestAnd:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(True, popxl.bool)
            b = popxl.variable(False, popxl.bool)
            c = ops.logical_and(a, b)
        assert len(g.tensors) == 3
        assert len(g.variables) == 2
        assert contains_op_of_type("And", _ir.op.AndOp, g)

    def test_needs_casting(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1, popxl.int32)
            b = popxl.variable(0, popxl.int32)
            c = ops.logical_and(a, b)
        assert len(g.tensors) == 5
        assert len(g.variables) == 2
        assert contains_op_of_type("And", _ir.op.AndOp, g)

    def test_dunder(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(True, popxl.bool)
            b = popxl.variable(False, popxl.bool)
            c = a & b
        assert len(g.tensors) == 3
        assert len(g.variables) == 2
        assert contains_op_of_type("And", _ir.op.AndOp, g)

    def test_ensure_tensor(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(True, popxl.bool)
            b = False
            c = a & b
        assert len(g.tensors) == 3
        assert len(g.variables) == 1
        assert len(g.constants) == 1
        assert contains_op_of_type("And", _ir.op.AndOp, g)

    def test_ensure_tensor_lhs(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = True
            b = popxl.variable(False, popxl.bool)
            c = a & b
        assert len(g.tensors) == 3
        assert len(g.variables) == 1
        assert len(g.constants) == 1
        assert contains_op_of_type("And", _ir.op.AndOp, g)
