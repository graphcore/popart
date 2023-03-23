# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl
import popxl.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestLog:
    def test_log_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones((1, 2, 3)))
            _ = ops.log(a)
        assert len(g.tensors) == 2
        assert contains_op_of_type("Log", _ir.op.LogOp, g)

    def test_log2_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones((1, 2, 3)))
            _ = ops.log2(a)
        assert len(g.tensors) == 5
        assert contains_op_of_type("Log", _ir.op.LogOp, g)
