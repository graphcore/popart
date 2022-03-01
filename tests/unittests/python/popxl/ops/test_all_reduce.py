# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import popxl
import popxl.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


def test_all_reduce_func():
    ir = popxl.Ir()
    g = ir.main_graph

    with g:
        x = popxl.variable(7)
        y = ops.collectives.all_reduce([x], ipus=[0])[0]

    assert x.shape == y.shape
    assert x.dtype == y.dtype
    assert len(g.tensors) == 2
    assert contains_op_of_type("AllReduce", _ir.op.collectives.AllReduceOp, g)


def test_all_reduce_identical_inputs_func():
    ir = popxl.Ir()
    g = ir.main_graph

    with g:
        x = popxl.variable(7)
        y = ops.collectives.all_reduce_identical_inputs([x], ipus=[0])[0]

    assert x.shape == y.shape
    assert x.dtype == y.dtype
    assert len(g.tensors) == 2
    assert contains_op_of_type("AllReduce", _ir.op.collectives.AllReduceOp, g)


def test_all_reduce_identical_grad_inputs_func():
    ir = popxl.Ir()
    g = ir.main_graph

    with g:
        x = popxl.variable(7)
        y = ops.collectives.all_reduce_identical_grad_inputs([x], ipus=[0])[0]

    assert x.shape == y.shape
    assert x.dtype == y.dtype
    assert len(g.tensors) == 2
    assert contains_op_of_type("AllReduce", _ir.op.collectives.AllReduceOp, g)
