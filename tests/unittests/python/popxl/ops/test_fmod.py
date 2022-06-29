# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops
from utils import contains_op_of_type


class TestFmod:
    """Ensure Fmod API is available. Including as __mod__ dunder on Tensor."""

    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            b = popxl.variable(2)
            _ = ops.fmod(a, b)
        assert len(g.tensors) == 3
        assert len(g.variables) == 2
        assert contains_op_of_type("Fmod", _ir.op.FmodOp, g)

    def test_dunder(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            b = popxl.variable(2)
            _ = a % b
        assert len(g.tensors) == 3
        assert len(g.variables) == 2
        assert contains_op_of_type("Fmod", _ir.op.FmodOp, g)

    def test_ensure_tensor(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            _ = a % 2
        assert len(g.tensors) == 3
        assert len(g.variables) == 1
        assert len(g.constants) == 1
        assert contains_op_of_type("Fmod", _ir.op.FmodOp, g)
