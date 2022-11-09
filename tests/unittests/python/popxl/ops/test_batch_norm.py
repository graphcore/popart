# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops
from utils import contains_op_of_type


class TestBatchNorm:
    def test_batch_norm(self):
        ir = popxl.Ir()
        g = ir.main_graph

        batch_size = 2
        channel_num = 4
        sp_size = 6

        with g:
            x = popxl.variable(4 * np.ones((batch_size, channel_num, sp_size)))
            scale = popxl.variable(np.ones(channel_num))
            bias = popxl.variable(np.zeros(channel_num))
            mean = popxl.variable(np.zeros(channel_num))
            var = popxl.variable(np.ones(channel_num))

            _ = ops.batch_norm_inference(x, scale, bias, mean, var)
        assert len(g.tensors) == 6
        assert len(g.variables) == 5
        assert contains_op_of_type("BatchNormalization", _ir.op.BatchNormOp, g)
