# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops
from popxl import dtypes
from utils import contains_op_of_type
import numpy as np


class TestDropout:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            x = popxl.variable(0)
            seed = popxl.variable(np.array([32, 32]), dtype=dtypes.uint32)
            c = ops.dropout(x, seed, 0.3)
        assert len(g.tensors) == 3
        assert len(g.variables) == 2
        assert contains_op_of_type("Dropout", _ir.op.DropoutOp, g)
