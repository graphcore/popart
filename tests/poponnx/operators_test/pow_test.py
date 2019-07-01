import numpy as np
import pytest
import poponnx
import torch
from op_tester import op_tester
import test_util as tu
from pprint import pprint


def test_pow(op_tester):
    d1 = np.random.randn(2).astype(np.float32)
    d2 = np.random.randn(2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.pow([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
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

    def reference(ref_data):
        t1 = torch.tensor(d1)
        t2 = torch.tensor(d2)
        out = torch.pow(t1, t2)

        return [out]

    # Need to have NaN == NaN to mirror numpy's functionality
    op_tester.equal_nan = True
    op_tester.run(init_builder, reference, step_type='infer')


def test_pow_grad(op_tester):
    # create test data
    d1 = np.random.randn(4, 1, 4).astype(np.float32)
    d2 = np.random.randn(3, 1).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.pow([i1, i2])
        builder.addOutputTensor(o)
        return [
            o,
            poponnx.reservedGradientPrefix() + i1,
            poponnx.reservedGradientPrefix() + i2,
            poponnx.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.tensor(d2, requires_grad=True)
        out = torch.pow(a, b)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, a.grad, b.grad, None]

    # Need to have NaN == NaN to mirror numpy's functionality
    op_tester.equal_nan = True
    op_tester.passes = ["PreUniRepl", "PowArg0GradOp", "PowArg1GradOp"]
    op_tester.run(init_builder, reference, 'train')
