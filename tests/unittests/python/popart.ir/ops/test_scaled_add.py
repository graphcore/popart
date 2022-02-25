# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import numpy as np
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops

from utils import contains_op_of_type


@pytest.mark.parametrize("inplace", [True, False])
class TestScaledAdd:
    def test_scaled_add_c_c(self, inplace):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            X = pir.variable(np.ones((2, 2)), dtype=pir.float32)
            Y = pir.variable(np.ones((2, 2)), dtype=pir.float32)
            if inplace:
                ops.scaled_add_(X, Y, a=0.9, b=0.1)
            else:
                ops.scaled_add(X, Y, a=0.9, b=0.1)
        assert len(g.tensors) == 3
        assert len(g.variables) == 2
        if inplace:
            assert contains_op_of_type("ScaledAddLhsInplace",
                                       _ir.op.ScaledAddLhsInplaceOp, g)
        else:
            assert contains_op_of_type("ScaledAdd", _ir.op.ScaledAddOp, g)

    def test_scaled_add_t_t(self, inplace):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            X = pir.variable(np.ones((2, 2)), dtype=pir.float32)
            Y = pir.variable(np.ones((2, 2)), dtype=pir.float32)
            if inplace:
                ops.scaled_add_(X, Y, a=pir.constant(0.9), b=pir.variable(0.1))
            else:
                ops.scaled_add(X, Y, a=pir.constant(0.9), b=pir.variable(0.1))
        assert len(g.tensors) == 5
        assert len(g.variables) == 3
        if inplace:
            assert contains_op_of_type("ScaledAddLhsInplace",
                                       _ir.op.ScaledAddLhsInplaceOp, g)
        else:
            assert contains_op_of_type("ScaledAdd", _ir.op.ScaledAddOp, g)

    def test_scaled_add_1_t(self, inplace):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            X = pir.variable(np.ones((2, 2)), dtype=pir.float32)
            Y = pir.variable(np.ones((2, 2)), dtype=pir.float32)
            if inplace:
                ops.scaled_add_(X, Y, b=pir.variable(0.1))
            else:
                ops.scaled_add(X, Y, b=pir.variable(0.1))
        assert len(g.tensors) == 4
        assert len(g.variables) == 3
        if inplace:
            assert contains_op_of_type("ScaledAddLhsInplace",
                                       _ir.op.ScaledAddLhsInplaceOp, g)
        else:
            assert contains_op_of_type("ScaledAdd", _ir.op.ScaledAddOp, g)
