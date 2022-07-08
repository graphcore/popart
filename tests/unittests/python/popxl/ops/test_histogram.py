# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops

from utils import contains_op_of_type


class TestHistogram:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones((1, 2, 3)))
            _ = ops.histogram(a, levels=[0.0001, 50000], absolute_of_input=True)
        assert len(g.tensors) == 2
        assert contains_op_of_type("Histogram", _ir.op.HistogramOp, g)
