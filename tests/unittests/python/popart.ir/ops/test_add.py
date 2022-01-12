# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
import pytest
from utils import contains_op_of_type


class TestAdd:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1)
            b = pir.variable(2)
            c = ops.add(a, b)
        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 2
        assert contains_op_of_type("Add", _ir.op.AddOp, g)

    def test_dunder(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1)
            b = pir.variable(2)
            c = a + b
        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 2
        assert contains_op_of_type("Add", _ir.op.AddOp, g)

    def test_ensure_tensor(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1)
            c = a + 2
        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 1
        assert len(g.get_constants()) == 1
        assert contains_op_of_type("Add", _ir.op.AddOp, g)

    def test_ensure_tensor_lhs(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1)
            c = 2 + a
        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 1
        assert len(g.get_constants()) == 1
        assert contains_op_of_type("Add", _ir.op.AddOp, g)

    def test_different_ipus(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1) + 0
            with pir.ipu(1):
                b = pir.variable(1) + 0

            with pytest.raises(ValueError):
                c = a + b

    def test_different_tile_sets(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1) + 0
            with pir.io_tiles():
                b = pir.variable(1) + 0

            with pytest.raises(ValueError):
                c = a + b
