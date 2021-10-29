# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops

from utils import contains_op_of_type


class TestIncrementMod:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1)
            c = ops.increment_mod(a, 1, 3)
        assert len(g.get_tensors()) == 2
        assert len(g.get_variables()) == 1
        assert contains_op_of_type("IncrementMod", _ir.op.IncrementModOp, g)

    def test_fn_(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1)
            c = ops.increment_mod_(a, 1, 3)
        assert len(g.get_tensors()) == 2
        assert len(g.get_variables()) == 1
        assert contains_op_of_type("IncrementModInplace",
                                   _ir.op.IncrementModInplaceOp, g)
