# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
from utils import contains_op_of_type


class TestSoftmax:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            a = pir.variable(np.ones((1, 2, 3)))
            c = ops.softmax(a, axis=1)
        assert len(g.tensors) == 2
        assert contains_op_of_type("Softmax", _ir.op.SoftmaxOp, g)

    def test_negative_axis(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            a = pir.variable(np.ones((1, 2, 3)))
            c = ops.softmax(a, axis=-1)
        assert len(g.tensors) == 2
        assert contains_op_of_type("Softmax", _ir.op.SoftmaxOp, g)
