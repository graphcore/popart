# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
from numpy.core.fromnumeric import size
import popart.ir as pir
import popart.ir.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestDynamicUpdateInplace:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            t = pir.variable(np.random.rand(3, 5, 7))
            index = pir.variable(np.array((1, 2)))
            axes = [0, 2]
            sizes = [1, 3]
            t_update = pir.variable(np.random.rand(sizes[0], sizes[1]))
            no_overlap = True
            c = ops.dynamic_update_inplace(t, index, t_update, axes, sizes,
                                           no_overlap)

        assert c.shape == t.shape
        assert len(g.get_tensors()) == 4
        assert contains_op_of_type("DynamicUpdateInplace",
                                   _ir.op.dynamic.DynamicUpdateInplaceOp, g)
