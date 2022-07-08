# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import popxl
import popxl.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestDynamicSlice:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            t = popxl.variable(np.random.rand(3, 5, 7))
            index = popxl.variable(np.array((1, 2)))
            axes = [0, 2]
            sizes = [1, 3]
            no_overlap = True
            c = ops.dynamic_slice(t, index, axes, sizes, no_overlap)

        assert c.shape == (sizes[0], t.shape[1], sizes[1])
        assert len(g.tensors) == 3
        assert contains_op_of_type("DynamicSlice", _ir.op.dynamic.DynamicSliceOp, g)
