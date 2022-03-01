# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops
from utils import contains_op_of_type


class TestSub:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            b = popxl.variable(2)
            c = ops.sub(a, b)
        assert len(g.tensors) == 3
        assert len(g.variables) == 2
        assert contains_op_of_type("Sub", _ir.op.SubtractOp, g)

    def test_dunder(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            b = popxl.variable(2)
            c = a - b
        assert len(ir.main_graph.tensors) == 3
        assert len(ir.main_graph.variables) == 2
        assert contains_op_of_type("Sub", _ir.op.SubtractOp, g)

    def test_ensure_tensor(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            c = a - 2
        assert len(ir.main_graph.tensors) == 3
        assert len(ir.main_graph.variables) == 1
        assert len(ir.main_graph.constants) == 1
        assert contains_op_of_type("Sub", _ir.op.SubtractOp, g)

    def test_ensure_tensor_lhs(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            c = 2 - a
        assert len(ir.main_graph.tensors) == 3
        assert len(ir.main_graph.variables) == 1
        assert len(ir.main_graph.constants) == 1
        assert contains_op_of_type("Sub", _ir.op.SubtractOp, g)
