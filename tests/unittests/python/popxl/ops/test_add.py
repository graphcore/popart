# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops
import pytest
from utils import contains_op_of_type


class TestAdd:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            b = popxl.variable(2)
            c = ops.add(a, b)
        assert len(g.tensors) == 3
        assert len(g.variables) == 2
        assert contains_op_of_type("Add", _ir.op.AddOp, g)

    def test_dunder(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            b = popxl.variable(2)
            c = a + b
        assert len(g.tensors) == 3
        assert len(g.variables) == 2
        assert contains_op_of_type("Add", _ir.op.AddOp, g)

    def test_fn_inplace(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            b = popxl.variable(2)
            c = ops.add_(a, b)
        assert len(g.tensors) == 3
        assert len(g.variables) == 2
        assert contains_op_of_type("AddLhsInplace", _ir.op.AddLhsInplaceOp, g)

    def test_inplace_dunder(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            b = popxl.variable(2)
            a += b
        assert len(g.tensors) == 3
        assert len(g.variables) == 2
        assert contains_op_of_type("AddLhsInplace", _ir.op.AddLhsInplaceOp, g)

    def test_ensure_tensor(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            c = a + 2
        assert len(g.tensors) == 3
        assert len(g.variables) == 1
        assert len(g.constants) == 1
        assert contains_op_of_type("Add", _ir.op.AddOp, g)

    def test_ensure_tensor_lhs(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            c = 2 + a
        assert len(g.tensors) == 3
        assert len(g.variables) == 1
        assert len(g.constants) == 1
        assert contains_op_of_type("Add", _ir.op.AddOp, g)

    def test_different_ipus(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1) + 0
            with popxl.ipu(1):
                b = popxl.variable(1) + 0

            with pytest.raises(ValueError):
                c = a + b

    def test_different_tile_sets(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1) + 0
            with popxl.io_tiles():
                b = popxl.variable(1) + 0

            with pytest.raises(ValueError):
                c = a + b
