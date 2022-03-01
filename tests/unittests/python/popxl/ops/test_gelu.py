# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl
import popxl.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestGelu:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones((1, 2, 3)))
            c = ops.gelu(a)
        assert len(g.tensors) == 2
        assert contains_op_of_type("Gelu", _ir.op.GeluOp, g)

    def test_inplace_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones((1, 2, 3)))
            c = ops.gelu_(a)
        assert len(g.tensors) == 2
        assert contains_op_of_type("GeluInplace", _ir.op.GeluInplaceOp, g)
