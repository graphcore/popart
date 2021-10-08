# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import numpy as np
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
from utils import contains_op_of_type


class TestGroupNorm:
    @pytest.mark.parametrize("groups", (1, 2, 4))
    def test_group_norm(self, groups):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            x = pir.variable(np.ones((2, 4)))
            weight = pir.variable(np.ones(4))
            bias = pir.variable(np.zeros(4))
            y = ops.group_norm(x, weight, bias, groups)
        assert len(g.get_tensors()) == 6
        assert len(g.get_variables()) == 3
        assert contains_op_of_type("GroupNorm", _ir.op.GroupNormOp, g)

    def test_layer_norm(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            x = pir.variable(np.ones((2, 4)))
            weight = pir.variable(np.ones(4))
            bias = pir.variable(np.zeros(4))
            y = ops.layer_norm(x, weight, bias)
        assert len(g.get_tensors()) == 6
        assert len(g.get_variables()) == 3
        assert contains_op_of_type("GroupNorm", _ir.op.GroupNormOp, g)
