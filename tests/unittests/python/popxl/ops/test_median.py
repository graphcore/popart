# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl
import popxl.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestMeadian:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones((1, 2, 3)))
            _ = ops.median(a, axis=[0, -1])
        assert len(g.tensors) == 3
        assert contains_op_of_type("ReduceMedian", _ir.op.ReduceMedianOp, g)
