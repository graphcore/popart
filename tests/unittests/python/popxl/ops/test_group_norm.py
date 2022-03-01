# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import numpy as np
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops
from utils import contains_op_of_type


class TestGroupNorm:
    @pytest.mark.parametrize("groups", (1, 2, 4))
    def test_group_norm(self, groups):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            x = popxl.variable(np.ones((2, 4)))
            weight = popxl.variable(np.ones(4))
            bias = popxl.variable(np.zeros(4))
            y = ops.group_norm(x, weight, bias, groups)
        assert len(g.tensors) == 6
        assert len(g.variables) == 3
        assert contains_op_of_type("GroupNormalization", _ir.op.GroupNormOp, g)

    def test_layer_norm(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            x = popxl.variable(np.ones((2, 4)))
            weight = popxl.variable(np.ones(4))
            bias = popxl.variable(np.zeros(4))
            y = ops.layer_norm(x, weight, bias)
        assert len(g.tensors) == 6
        assert len(g.variables) == 3
        assert contains_op_of_type("GroupNormalization", _ir.op.GroupNormOp, g)
