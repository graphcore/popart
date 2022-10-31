# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import List, Tuple, Union
import pytest
import numpy as np

import popxl
import popxl.ops as ops
import popart._internal.ir as _ir

from utils import contains_op_of_type


class TestSubsample:
    @pytest.mark.parametrize(
        "input_shape, slices",
        [
            ((10, 2), [2, 1]),
            ((10, 2), [2]),
            ((10,), 2),
        ],
    )
    def test_fn(self, input_shape: Tuple[int, ...], slices: Union[int, List[int]]):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones(input_shape))
            _ = ops.subsample(a, slices)
            assert len(g.tensors) == 2
            assert contains_op_of_type("Subsample", _ir.op.SubsampleOp, g)
