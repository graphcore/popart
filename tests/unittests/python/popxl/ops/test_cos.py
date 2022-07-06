# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops
import numpy as np
from utils import contains_op_of_type


class TestCos:
    def test_cos(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.random.rand(2, 2))
            _ = ops.cos(a)
        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        assert contains_op_of_type("Cos", _ir.op.CosOp, g)
