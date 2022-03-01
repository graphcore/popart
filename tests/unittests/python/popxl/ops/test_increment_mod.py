# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops

from utils import contains_op_of_type


class TestIncrementMod:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            c = ops.increment_mod(a, 1, 3)
        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        assert contains_op_of_type("IncrementMod", _ir.op.IncrementModOp, g)

    def test_fn_(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            c = ops.increment_mod_(a, 1, 3)
        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        assert contains_op_of_type("IncrementModInplace",
                                   _ir.op.IncrementModInplaceOp, g)
