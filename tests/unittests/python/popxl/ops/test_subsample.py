# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import List, Tuple, Union
import pytest
import numpy as np
from contextlib import suppress

import popxl
import popxl.ops as ops
import popart._internal.ir as _ir

from utils import contains_op_of_type


class TestSubsample:
    @pytest.mark.parametrize(
        "input_shape, slices, expect_fail",
        [
            ((10, 2), [2, 1], False),
            ((10, 2), [2], True),
            ((10,), 2, False),
        ],
    )
    def test_fn(
        self,
        input_shape: Tuple[int, ...],
        slices: Union[int, List[int]],
        expect_fail: bool,
    ):
        ir = popxl.Ir()
        g = ir.main_graph

        ctx = pytest.raises(ValueError) if expect_fail else suppress()

        with ctx, g:
            a = popxl.variable(np.ones(input_shape))
            _ = ops.subsample(a, slices)
            assert len(g.tensors) == 2
            assert contains_op_of_type("Subsample", _ir.op.SubsampleOp, g)
