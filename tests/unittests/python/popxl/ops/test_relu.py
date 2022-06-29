# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl
import popxl.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestRelu:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones((1, 2, 3)))
            _ = ops.relu(a)
        assert len(g.tensors) == 2
        assert contains_op_of_type("Relu", _ir.op.ReluOp, g)

    def test_inplace_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones((1, 2, 3)))
            _ = ops.relu_(a)
        assert len(g.tensors) == 2
        assert contains_op_of_type("ReluInplace", _ir.op.ReluInplaceOp, g)
