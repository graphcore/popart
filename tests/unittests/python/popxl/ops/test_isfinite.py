# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl
import popxl.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestFinite:
    def test_fn_isinf(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones((1, 2, 3)))
            _ = ops.isfinite(a)
        assert len(g.tensors) == 6
        assert contains_op_of_type("IsInf", _ir.op.IsInf, g)
        assert contains_op_of_type("IsNaN", _ir.op.IsNaN, g)
