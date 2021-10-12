# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
import numpy as np
from utils import contains_op_of_type


class TestDetach:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(np.ones((1, 2, 3)))
            c = ops.detach(a)
        assert len(g.get_tensors()) == 2
        assert len(g.get_variables()) == 1
        assert contains_op_of_type("Detach", _ir.op.DetachOp, g)

    def test_dunder(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(np.ones((1, 2, 3)))
            c = a.detach()
        assert len(g.get_tensors()) == 2
        assert len(g.get_variables()) == 1
        assert contains_op_of_type("Detach", _ir.op.DetachOp, g)
