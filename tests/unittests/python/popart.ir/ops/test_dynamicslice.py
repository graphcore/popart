# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestDynamicSlice:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            t = pir.variable(np.random.rand(3, 5, 7))
            index = pir.variable(np.array((1, 2)))
            axes = [0, 2]
            sizes = [1, 3]
            noOverlap = True
            c = ops.dynamicslice(t, index, axes, sizes, noOverlap)

        assert c.shape == (sizes[0], t.shape[1], sizes[1])
        assert len(g.get_tensors()) == 3
        assert contains_op_of_type("DynamicSlice",
                                   _ir.op.dynamic.DynamicSliceOp, g)
