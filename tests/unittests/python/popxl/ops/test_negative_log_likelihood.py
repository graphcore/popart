# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops
import pytest

from utils import contains_op_of_type


class TestNll:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.zeros((2, 10)), popxl.float32)
            b = popxl.variable(np.zeros((2)), popxl.int32)
            _ = ops.nll_loss(a, b)
        assert len(g.tensors) == 3
        assert contains_op_of_type("Nll", _ir.op.NllOp, g)

    def test_ignore_index(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.zeros((2, 10)), popxl.float32)
            b = popxl.variable(np.zeros((2)), popxl.int32)
            _ = ops.nll_loss(a, b, ignore_index=5)
        assert len(g.tensors) == 3
        assert contains_op_of_type("Nll", _ir.op.NllOp, g)

    def test_reduction_type_error(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.zeros((2, 10)), popxl.float32)
            b = popxl.variable(np.zeros((2)), popxl.int32)
            with pytest.raises(ValueError):
                _ = ops.nll_loss(a, b, reduction='foo')


class TestNllWithSoftmaxGrad:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.zeros((2, 10)), popxl.float32)
            b = popxl.variable(np.zeros((2)), popxl.float32)
            _ = ops.nll_loss_with_softmax_grad(a, b)
        assert len(g.tensors) == 5
        assert contains_op_of_type("NlllWithSoftmaxGradDirect",
                                   _ir.op.NlllWithSoftmaxGradDirectOp, g)

    def test_loss_grad(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.zeros((2, 10)), popxl.float32)
            b = popxl.variable(np.zeros((2)), popxl.float32)
            _ = ops.nll_loss_with_softmax_grad(a, b, loss_grad=2)
        assert len(g.tensors) == 5
        assert contains_op_of_type("NlllWithSoftmaxGradDirect",
                                   _ir.op.NlllWithSoftmaxGradDirectOp, g)

    def test_ignore_index(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.zeros((2, 10)), popxl.float32)
            b = popxl.variable(np.zeros((2)), popxl.float32)
            _ = ops.nll_loss_with_softmax_grad(a, b, ignore_index=5)
        assert len(g.tensors) == 5
        assert contains_op_of_type("NlllWithSoftmaxGradDirect",
                                   _ir.op.NlllWithSoftmaxGradDirectOp, g)

    def test_reduction_type_error(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.zeros((2, 10)), popxl.float32)
            b = popxl.variable(np.zeros((2)), popxl.int32)
            with pytest.raises(ValueError):
                _ = ops.nll_loss_with_softmax_grad(a, b, reduction='foo')
