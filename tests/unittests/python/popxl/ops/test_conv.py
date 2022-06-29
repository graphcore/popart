# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np
import pytest
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestConv:
    @pytest.mark.parametrize("pad_type",
                             ('not_set', 'same_upper', 'same_lower', 'valid'))
    def test_fn(self, pad_type):
        batch_size = 1
        in_channel = 4
        out_channel = 4
        height = 1
        width = 1
        h_kernel = 1
        w_kernel = 1
        strides = (1, 1)
        pads = (0, 0, 0, 0)
        dilations = (1, 1)
        group = 1
        ir = popxl.Ir()
        g = ir.main_graph
        with ir.main_graph:
            t = popxl.variable(
                np.random.rand(batch_size, in_channel, height, width))
            weight = popxl.variable(
                np.random.rand(out_channel, int(in_channel / group), h_kernel,
                               w_kernel))
            _ = ops.conv(t, weight, strides, pads, dilations, group, pad_type)
        assert len(g.tensors) == 3
        assert len(g.variables) == 2
        assert contains_op_of_type("Conv", _ir.op.ConvOp, g)
