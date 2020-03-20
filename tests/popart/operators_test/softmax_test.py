# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import torch
from op_tester import op_tester
import popart


def test_softmax(op_tester):
    # create test data
    d1 = np.random.rand(1, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.softmax([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        # 'dim' corresponds to dim index over which
        # to perform softmax
        lsm = torch.nn.Softmax(dim=1)
        b = lsm(a)
        return [b]

    op_tester.run(init_builder, reference, 'infer')


def test_softmax_grad(op_tester):
    # create test data
    d1 = np.random.rand(1, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.softmax([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        sm = torch.nn.Softmax(dim=1)
        b = sm(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'train')


def test_softmax_rank1_axis0(op_tester):
    d1 = np.random.rand(5).astype(np.float32)
    _test_softmax(op_tester, d1, 0)


def test_softmax_rank3_axis0(op_tester):
    d1 = np.random.rand(3, 4, 5).astype(np.float32)
    _test_softmax(op_tester, d1, 0)


def test_softmax_rank3_axis1(op_tester):
    d1 = np.random.rand(3, 4, 5).astype(np.float32)
    _test_softmax(op_tester, d1, 1)


def test_softmax_rank3_axis2(op_tester):
    d1 = np.random.rand(3, 4, 5).astype(np.float32)
    _test_softmax(op_tester, d1, 2)


def test_softmax_rank1_axis0_grad(op_tester):
    d1 = np.random.rand(5).astype(np.float32)
    _test_softmax_grad(op_tester, d1, 0)


def test_softmax_rank3_axis0_grad(op_tester):
    d1 = np.random.rand(3, 4, 5).astype(np.float32)
    _test_softmax_grad(op_tester, d1, 0)


def test_softmax_rank3_axis1_grad(op_tester):
    d1 = np.random.rand(3, 4, 5).astype(np.float32)
    _test_softmax_grad(op_tester, d1, 1)


def test_softmax_rank3_axis2_grad(op_tester):
    d1 = np.random.rand(3, 4, 5).astype(np.float32)
    _test_softmax_grad(op_tester, d1, 2)


def _test_softmax(op_tester, data, axis):
    def init_builder(builder):
        i1 = builder.addInputTensor(data)
        o = builder.aiOnnx.softmax([i1], axis)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        n = 1
        for i in data.shape[:axis]:
            n *= i
        d = 1
        for i in data.shape[axis:]:
            d *= i

        a = torch.tensor(data.reshape(n, d), requires_grad=True)
        b = a.view(n, d)
        lsm = torch.nn.Softmax(dim=1)
        c = lsm(b)
        o = c.view(*data.shape)
        return [o]

    op_tester.run(init_builder, reference, 'infer')


def _test_softmax_grad(op_tester, data, axis):
    def init_builder(builder):
        i1 = builder.addInputTensor(data)
        o = builder.aiOnnx.softmax([i1], axis)
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        n = 1
        for i in data.shape[:axis]:
            n *= i
        d = 1
        for i in data.shape[axis:]:
            d *= i

        torch_data = torch.tensor(data, requires_grad=True)
        reshaped_data = torch_data.view(n, d)
        sm = torch.nn.Softmax(dim=1)
        softmax_out = sm(reshaped_data)
        reshaped_softmax_out = softmax_out.view(*data.shape)

        d__o = ref_data.getOutputTensorGrad(0)
        reshaped_softmax_out.backward(torch.tensor(d__o))

        return [reshaped_softmax_out, torch_data.grad, None]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'train')
