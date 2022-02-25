# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops

from utils import contains_op_of_type


class TestNll:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            a = pir.variable(np.zeros((2, 10)), pir.float32)
            b = pir.variable(np.zeros((2)), pir.float32)
            c = ops.nll_loss_with_softmax_grad(a, b)
        assert len(g.tensors) == 5
        assert contains_op_of_type("NlllWithSoftmaxGradDirect",
                                   _ir.op.NlllWithSoftmaxGradDirectOp, g)

    def test_loss_grad(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            a = pir.variable(np.zeros((2, 10)), pir.float32)
            b = pir.variable(np.zeros((2)), pir.float32)
            c = ops.nll_loss_with_softmax_grad(a, b, 2)
        assert len(g.tensors) == 5
        assert contains_op_of_type("NlllWithSoftmaxGradDirect",
                                   _ir.op.NlllWithSoftmaxGradDirectOp, g)

    def test_ignore_index(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            a = pir.variable(np.zeros((2, 10)), pir.float32)
            b = pir.variable(np.zeros((2)), pir.float32)
            c = ops.nll_loss_with_softmax_grad(a, b, ignore_index=5)
        assert len(g.tensors) == 5
        assert contains_op_of_type("NlllWithSoftmaxGradDirect",
                                   _ir.op.NlllWithSoftmaxGradDirectOp, g)
