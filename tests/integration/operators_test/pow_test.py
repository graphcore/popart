# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
import pytest


def test_pow(op_tester):
    d1 = np.random.randn(2).astype(np.float32)
    d2 = np.random.randn(2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.pow([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        t1 = torch.tensor(d1)
        t2 = torch.tensor(d2)
        out = torch.pow(t1, t2)

        return [out]

    # Need to have NaN == NaN to mirror numpy's functionality
    op_tester.equal_nan = True
    op_tester.run(init_builder, reference, step_type='infer')


def test_broadcast_pow(op_tester):
    d1 = np.random.randn(2, 2).astype(np.float32)
    d2 = np.random.randn(2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.pow([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        t1 = torch.tensor(d1)
        t2 = torch.tensor(d2)
        out = torch.pow(t1, t2)

        return [out]

    # Need to have NaN == NaN to mirror numpy's functionality
    op_tester.equal_nan = True
    op_tester.run(init_builder, reference, step_type='infer')


@pytest.mark.parametrize("precision", [np.float32, np.float16])
def test_pow_grad(op_tester, precision):
    # create test data
    d1 = np.random.randn(4, 1, 4)
    d2 = np.random.randn(3, 1)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1.astype(precision))
        i2 = builder.addInputTensor(d2.astype(precision))
        o = builder.aiOnnx.pow([i1, i2])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.tensor(d2, requires_grad=True)
        out = torch.pow(a, b)

        d__o = torch.tensor(ref_data.getOutputTensorGrad(0)).to(torch.float32)
        assert not torch.isnan(d__o).any()
        out.backward(d__o)
        return_type = torch.float16 if precision == np.float16 else torch.float32
        return [
            out.to(return_type),
            a.grad.to(return_type),
            b.grad.to(return_type), None
        ]

    # Need to have NaN == NaN to mirror numpy's functionality
    if precision == np.float16:
        op_tester.atol = 2e-03
    op_tester.equal_nan = True
    op_tester.setPatterns(["PreUniRepl", "PowArg0GradOp", "PowArg1GradOp"],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')
