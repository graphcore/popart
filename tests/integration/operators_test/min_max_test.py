# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
import pytest

from op_tester import op_tester


def test_max_training(op_tester):
    d1 = np.random.rand(5, 7, 5).astype(np.float32)
    d2 = np.random.rand(7, 5).astype(np.float32)
    d3 = np.random.rand(5).astype(np.float32)
    d4 = np.random.rand(1, 1, 5).astype(np.float32)
    d5 = np.random.rand(5, 1, 5).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d4)
        i5 = builder.addInputTensor(d5)
        o = builder.aiOnnx.max([i1, i2, i3, i4, i5], "test_max")
        builder.addOutputTensor(o)
        return [
            o,
            popart.TensorId(popart.reservedGradientPrefix() + i1),
            popart.TensorId(popart.reservedGradientPrefix() + i2),
            popart.TensorId(popart.reservedGradientPrefix() + i3),
            popart.TensorId(popart.reservedGradientPrefix() + i4),
            popart.TensorId(popart.reservedGradientPrefix() + i5),
            popart.TensorId(popart.reservedGradientPrefix() + o)
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        t3 = torch.tensor(d3, requires_grad=True)
        t4 = torch.tensor(d4, requires_grad=True)
        t5 = torch.tensor(d5, requires_grad=True)

        out = torch.max(t1, t2)
        out = torch.max(t3, out)
        out = torch.max(t4, out)
        out = torch.max(t5, out)

        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))

        return [out, t1.grad, t2.grad, t3.grad, t4.grad, t5.grad, d__o]

    op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_min_training_0(op_tester):
    d1 = np.random.rand(3, 4).astype(np.float32)
    d2 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.min([i1, i2], "test_min")
        builder.addOutputTensor(o)
        return [
            o,
            popart.TensorId(popart.reservedGradientPrefix() + i1),
            popart.TensorId(popart.reservedGradientPrefix() + i2),
            popart.TensorId(popart.reservedGradientPrefix() + o)
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)

        out = torch.min(t1, t2)

        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))

        return [out, t1.grad, t2.grad, d__o]

    op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_min_training_1(op_tester):
    d1 = np.random.rand(2, 3, 4).astype(np.float32)
    d2 = np.random.rand(4).astype(np.float32)
    d3 = np.random.rand(1, 1, 4).astype(np.float32)
    d4 = np.random.rand(2, 1, 4).astype(np.float32)
    d5 = np.random.rand(1, 3, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d4)
        i5 = builder.addInputTensor(d5)
        o = builder.aiOnnx.min([i1, i2, i3, i4, i5], "test_min")
        builder.addOutputTensor(o)
        return [
            o,
            popart.TensorId(popart.reservedGradientPrefix() + i1),
            popart.TensorId(popart.reservedGradientPrefix() + i2),
            popart.TensorId(popart.reservedGradientPrefix() + i3),
            popart.TensorId(popart.reservedGradientPrefix() + i4),
            popart.TensorId(popart.reservedGradientPrefix() + i5),
            popart.TensorId(popart.reservedGradientPrefix() + o)
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        t3 = torch.tensor(d3, requires_grad=True)
        t4 = torch.tensor(d4, requires_grad=True)
        t5 = torch.tensor(d5, requires_grad=True)

        out = torch.min(t1, t2)
        out = torch.min(t3, out)
        out = torch.min(t4, out)
        out = torch.min(t5, out)

        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))

        return [out, t1.grad, t2.grad, t3.grad, t4.grad, t5.grad, d__o]

    op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')
