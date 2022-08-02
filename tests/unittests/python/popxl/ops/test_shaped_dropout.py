# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops
from popxl import dtypes
from utils import contains_op_of_type
import numpy as np


class TestShapedDropout:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            t = popxl.variable(np.random.rand(8, 2))
            seed = popxl.variable(np.array([32, 32]), dtype=dtypes.uint32)
            _ = ops.shaped_dropout(t, seed, [8, 1], 0.5)
        assert len(g.tensors) == 3
        assert len(g.variables) == 2
        assert contains_op_of_type("ShapedDropout", _ir.op.ShapedDropoutOp, g)
