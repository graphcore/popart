# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestOneHot:
    def test_fn(self):
        dim = 1
        ir = popxl.Ir()
        g = ir.main_graph
        with ir.main_graph:
            t = popxl.variable(np.array([[1, 5]]).astype(np.int32))
            num_classes = popxl.constant(np.array(6).astype(np.int32))
            values = popxl.constant(np.array([0, 1]).astype(np.int32))
            res = ops.onehot(t, num_classes, values, dim)
        assert len(g.tensors) == 4
        assert len(g.variables) == 1
        assert contains_op_of_type("OneHot", _ir.op.OnehotOp, g)
