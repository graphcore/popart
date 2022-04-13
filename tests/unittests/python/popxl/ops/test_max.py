# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl
import popxl.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestMax:
    def test_reduce_max(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones((1, 2, 3)))
            c = ops.max(a, axis=[0, -1])
        assert len(g.tensors) == 2
        assert contains_op_of_type("ReduceMax", _ir.op.ReduceMaxOp, g)

    def test_element_wise_max(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            b = popxl.variable(2)
            c = ops.maximum(a, b)
        assert len(g.tensors) == 3
        assert len(g.variables) == 2
        assert contains_op_of_type("Max", _ir.op.MaxOp, g)
