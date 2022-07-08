# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops
from utils import contains_op_of_type


class TestRoiAlign:
    """Ensure RoiAlign API is available and raises exceptions when expected"""

    def test_fn(self):
        output_size = (6, 6)
        spatial_scale = 0.05
        sampling_ratio = 2
        batch_size = 2
        channel = 4
        width = 50
        height = 50
        num_roi = 256
        ir = popxl.Ir()
        g = ir.main_graph
        with g:
            a = popxl.variable(np.random.randn(batch_size, channel, width, height))
            b = popxl.variable(np.random.randn(num_roi, 4))
            c = popxl.variable(
                np.random.randint(0, batch_size, (num_roi), dtype=np.int32)
            )
            _ = ops.roi_align(a, b, c, output_size, spatial_scale, sampling_ratio)
        assert contains_op_of_type("RoiAlign", _ir.op.RoiAlignOp, g)
