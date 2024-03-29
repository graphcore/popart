# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import popxl
import popxl.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestDynamicUpdate:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            t = popxl.variable(np.random.rand(3, 5, 7))
            index = popxl.variable(np.array((1, 2)))
            axes = [0, 2]
            sizes = [1, 3]
            t_update = popxl.variable(np.random.rand(sizes[0], sizes[1]))
            no_overlap = True
            c = ops.dynamic_update(t, index, t_update, axes, sizes, no_overlap)

        assert c.shape == t.shape
        assert len(g.tensors) == 4
        assert contains_op_of_type("DynamicUpdate", _ir.op.dynamic.DynamicUpdateOp, g)

    def test_fn_inplace(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            t = popxl.variable(np.random.rand(3, 5, 7))
            index = popxl.variable(np.array((1, 2)))
            axes = [0, 2]
            sizes = [1, 3]
            t_update = popxl.variable(np.random.rand(sizes[0], sizes[1]))
            no_overlap = True
            c = ops.dynamic_update_(t, index, t_update, axes, sizes, no_overlap)

        assert c.shape == t.shape
        assert len(g.tensors) == 4
        assert contains_op_of_type(
            "DynamicUpdateInplace", _ir.op.dynamic.DynamicUpdateInplaceOp, g
        )
