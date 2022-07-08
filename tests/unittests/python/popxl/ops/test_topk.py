# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestTopK:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph
        with ir.main_graph:
            t = popxl.variable(np.random.rand(2, 10).astype("float32"))
            k = 2
            axis = 1
            largest = True
            sorted_ = True
            _ = ops.topk(t, k, axis, largest, sorted_)
        assert len(g.tensors) == 3
        assert len(g.variables) == 1
        assert contains_op_of_type("TopK", _ir.op.TopKOp, g)
