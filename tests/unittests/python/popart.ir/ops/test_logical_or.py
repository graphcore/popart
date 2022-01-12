# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
from utils import contains_op_of_type


class TestOr:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(True, pir.bool)
            b = pir.variable(False, pir.bool)
            c = ops.logical_or(a, b)
        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 2
        assert contains_op_of_type("Or", _ir.op.OrOp, g)

    def test_needs_casting(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1, pir.int32)
            b = pir.variable(0, pir.int32)
            c = ops.logical_or(a, b)
        assert len(g.get_tensors()) == 5
        assert len(g.get_variables()) == 2
        assert contains_op_of_type("Or", _ir.op.OrOp, g)

    def test_dunder(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(True, pir.bool)
            b = pir.variable(False, pir.bool)
            c = a | b
        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 2
        assert contains_op_of_type("Or", _ir.op.OrOp, g)

    def test_ensure_tensor(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(True, pir.bool)
            b = False
            c = a | b
        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 1
        assert len(g.get_constants()) == 1
        assert contains_op_of_type("Or", _ir.op.OrOp, g)

    def test_ensure_tensor_lhs(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = True
            b = pir.variable(False, pir.bool)
            c = a | b
        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 1
        assert len(g.get_constants()) == 1
        assert contains_op_of_type("Or", _ir.op.OrOp, g)
