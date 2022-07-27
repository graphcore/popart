# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np
import popart._internal.ir as _ir
from utils import contains_op_of_type


def test_cumsum():
    dim = 0
    ir = popxl.Ir()
    g = ir.main_graph

    with ir.main_graph:
        t = popxl.variable(np.random.rand(5, 10, 20))
        _ = ops.cumsum(t, dim=dim)

    assert len(g.tensors) == 3
    assert len(g.variables) == 1
    assert contains_op_of_type("CumSum", _ir.op.CumSumOp, g)
