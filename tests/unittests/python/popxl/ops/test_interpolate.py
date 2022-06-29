# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestInterpolate:
    def test_fn(self):
        batch_size = 1
        in_channel = 4
        height = 5
        width = 5
        ir = popxl.Ir()
        g = ir.main_graph
        with ir.main_graph:
            t = popxl.variable(
                np.random.rand(batch_size, in_channel, height, width))
            _ = ops.interpolate(t, scale_factor=(1, 1, 1, 1), mode='nearest')
        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        assert contains_op_of_type("Resize", _ir.op.ResizeOp, g)
