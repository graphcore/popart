# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest

import numpy as np
import popart._internal.ir as _ir
import popxl
from popxl import Tensor
import popxl.ops as ops

from utils import contains_op_of_type


@pytest.mark.parametrize("exp_type", [Tensor, float, int])
def test_pow_op(exp_type):
    ir = popxl.Ir()
    g = ir.main_graph

    with g:
        if exp_type is Tensor:
            X = popxl.variable(np.ones((2, 2)), dtype=popxl.float32)
            Y = popxl.variable(np.ones((2, 2)), dtype=popxl.float32)
        elif exp_type is float:
            X = popxl.variable(np.ones((2, 2)), dtype=popxl.float32)
            Y = 1.0
        elif exp_type is int:
            X = popxl.variable(np.ones((2, 2)), dtype=popxl.int32)
            Y = 1
        ops.pow(X, Y)

    assert len(g.tensors) == 3
    assert len(g.variables) == (2 if exp_type is Tensor else 1)
    assert contains_op_of_type("Pow", _ir.op.PowOp, g)


@pytest.mark.parametrize("exp_type", [Tensor, float, int])
def test_pow_tensor(exp_type):
    ir = popxl.Ir()
    g = ir.main_graph

    with g:
        if exp_type is Tensor:
            X = popxl.variable(np.ones((2, 2)), dtype=popxl.float32)
            Y = popxl.variable(np.ones((2, 2)), dtype=popxl.float32)
        elif exp_type is float:
            X = popxl.variable(np.ones((2, 2)), dtype=popxl.float32)
            Y = 1.0
        elif exp_type is int:
            X = popxl.variable(np.ones((2, 2)), dtype=popxl.int32)
            Y = 1
        _ = X ** Y
    assert len(g.tensors) == 3

    assert len(g.variables) == (2 if exp_type is Tensor else 1)
    assert contains_op_of_type("Pow", _ir.op.PowOp, g)


def test_rpow_tensor():
    ir = popxl.Ir()
    g = ir.main_graph

    with g:
        X = np.ones((2, 2))
        Y = popxl.variable(np.ones((2, 2)), dtype=popxl.float32)
        _ = X ** Y

    assert len(g.tensors) == 3
    assert len(g.variables) == 1
    assert contains_op_of_type("Pow", _ir.op.PowOp, g)
