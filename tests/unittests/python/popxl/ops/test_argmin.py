# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestArgMin:
    def test_fn(self):
        dim = 0
        keepdim = False
        ir = popxl.Ir()
        g = ir.main_graph
        with ir.main_graph:
            t = popxl.variable(np.random.rand(5, 10, 20))
            _ = ops.argmin(t, dim, keepdim)
        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        assert contains_op_of_type("ArgMin", _ir.op.ArgMinOp, g)
