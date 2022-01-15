# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


def test_all_reduce_func():
    ir = pir.Ir()
    g = ir.main_graph()

    with g:
        x = pir.variable(7)
        y = ops.collectives.all_reduce([x], ipus=[0])[0]

    assert x.shape == y.shape
    assert x.dtype == y.dtype
    assert len(g.get_tensors()) == 2
    assert contains_op_of_type("AllReduce", _ir.op.collectives.AllReduceOp, g)


def test_all_reduce_identical_inputs_func():
    ir = pir.Ir()
    g = ir.main_graph()

    with g:
        x = pir.variable(7)
        y = ops.collectives.all_reduce_identical_inputs([x], ipus=[0])[0]

    assert x.shape == y.shape
    assert x.dtype == y.dtype
    assert len(g.get_tensors()) == 2
    assert contains_op_of_type("AllReduce", _ir.op.collectives.AllReduceOp, g)


def test_all_reduce_identical_grad_inputs_func():
    ir = pir.Ir()
    g = ir.main_graph()

    with g:
        x = pir.variable(7)
        y = ops.collectives.all_reduce_identical_grad_inputs([x], ipus=[0])[0]

    assert x.shape == y.shape
    assert x.dtype == y.dtype
    assert len(g.get_tensors()) == 2
    assert contains_op_of_type("AllReduce", _ir.op.collectives.AllReduceOp, g)
