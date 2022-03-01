# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import numpy as np
import popxl
import popxl.ops as ops
from utils import contains_op_of_type


class TestSoftmax:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones((1, 2, 3)))
            c = ops.softmax(a, axis=1)
        assert len(g.tensors) == 2
        assert contains_op_of_type("Softmax", _ir.op.SoftmaxOp, g)

    def test_negative_axis(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones((1, 2, 3)))
            c = ops.softmax(a, axis=-1)
        assert len(g.tensors) == 2
        assert contains_op_of_type("Softmax", _ir.op.SoftmaxOp, g)
