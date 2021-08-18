# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import torch
from op_tester import op_tester
import test_util as tu
from pprint import pprint


def test_sub(op_tester):
    # create test data
    d1 = np.random.randn(4).astype(np.float32)
    d2 = np.random.randn(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.sub([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.tensor(d2, requires_grad=True)
        out = a - b

        return [out]

    # Need to have NaN == NaN to mirror numpy's functionality
    op_tester.equal_nan = True
    op_tester.run(init_builder, reference, step_type='infer')


def test_broadcast_sub(op_tester):
    # create test data
    d1 = np.random.randn(4, 18).astype(np.float32)
    d2 = np.random.randn(4, 1).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.sub([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.tensor(d2, requires_grad=True)
        out = a - b

        return [out]

    # Need to have NaN == NaN to mirror numpy's functionality
    op_tester.equal_nan = True
    op_tester.run(init_builder, reference, step_type='infer')


def test_sub_grad(op_tester):
    # create test data
    d1 = np.random.randn(4).astype(np.float32)
    d2 = np.random.randn(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.sub([i1, i2])
        builder.addOutputTensor(o)
        return [
            o,
            popart.TensorId(popart.reservedGradientPrefix() + i1),
            popart.TensorId(popart.reservedGradientPrefix() + i2),
            popart.TensorId(popart.reservedGradientPrefix() + o)
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.tensor(d2, requires_grad=True)
        out = a - b

        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        assert not torch.isnan(d__o).any()
        out.backward(d__o)

        return [out, a.grad, b.grad, None]

    # Need to have NaN == NaN to mirror numpy's functionality
    op_tester.equal_nan = True
    op_tester.setPatterns(["SubtractArg1GradOp"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_broadcast_sub_grad(op_tester):
    # create test data
    d1 = np.random.randn(4, 18).astype(np.float32)
    d2 = np.random.randn(4, 1).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.sub([i1, i2])
        builder.addOutputTensor(o)
        return [
            o,
            popart.TensorId(popart.reservedGradientPrefix() + i1),
            popart.TensorId(popart.reservedGradientPrefix() + i2),
            popart.TensorId(popart.reservedGradientPrefix() + o)
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.tensor(d2, requires_grad=True)
        out = a - b

        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        assert not torch.isnan(d__o).any()
        out.backward(d__o)

        return [out, a.grad, b.grad, None]

    # Need to have NaN == NaN to mirror numpy's functionality
    op_tester.equal_nan = True
    op_tester.setPatterns(["SubtractArg1GradOp"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')
