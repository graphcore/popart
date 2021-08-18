# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest
import torch
from op_tester import op_tester


def logsoftmax_reference(data, axis):
    # ONNX logsoftmax does not match the torch logsoftmax (fixed in opset 13)
    n = 1
    for i in data.shape[:axis]:
        n *= i
    d = 1
    for i in data.shape[axis:]:
        d *= i

    data_2d = data.reshape(n, d)
    lsm = torch.nn.LogSoftmax(dim=1)
    out = lsm(data_2d)
    return out.view(*data.shape)


@pytest.mark.parametrize("inplace", [True, False])
def test_logsoftmax(op_tester, inplace):
    d1 = np.random.rand(1, 4).astype(np.float32)
    _test_logsoftmax(op_tester, d1, inplace=inplace)


def test_logsoftmax_stability(op_tester):
    d1 = np.array([10, 100, 1000], dtype=np.float32).reshape(1, 3)
    _test_logsoftmax(op_tester, d1)


def test_logsoftmax_negative_axis(op_tester):
    d1 = np.random.rand(1, 4).astype(np.float32)
    _test_logsoftmax(op_tester, d1, axis=-1)


def test_logsoftmax_large_number(op_tester):
    x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]],
                 dtype=np.float32)
    _test_logsoftmax(op_tester, x)


@pytest.mark.parametrize("axis", range(3))
def test_logsoftmax_axis(op_tester, axis):
    np.random.seed(0)
    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    _test_logsoftmax(op_tester, x, axis=axis)


@pytest.mark.parametrize("inplace", [True, False])
def test_logsoftmax_grad(op_tester, inplace):
    # create test data
    d1 = np.random.rand(1, 10).astype(np.float32)
    _test_logsoftmax_grad(op_tester, d1, inplace=inplace)


def test_logsoftmax_grad_large_number(op_tester):
    x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]],
                 dtype=np.float32)
    _test_logsoftmax_grad(op_tester, x)


@pytest.mark.parametrize("axis", range(3))
def test_logsoftmax_grad_axis(op_tester, axis):
    np.random.seed(0)
    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    _test_logsoftmax_grad(op_tester, x, axis=axis)


def _test_logsoftmax(op_tester, data, axis=1, inplace=True):
    def init_builder(builder):
        i1 = builder.addInputTensor(data)
        o = builder.aiOnnx.logsoftmax([i1], axis=axis)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        return [logsoftmax_reference(torch.tensor(data), axis)]

    op_tester.setPatterns([], enableRuntimeAsserts=False)
    op_tester.inplacing = inplace
    op_tester.run(init_builder, reference, 'infer')


def _test_logsoftmax_grad(op_tester, data, axis=1, inplace=True):
    def init_builder(builder):
        i1 = builder.addInputTensor(data)
        o = builder.aiOnnx.logsoftmax([i1], axis=axis)
        builder.addOutputTensor(o)
        return [
            o,
            popart.TensorId(popart.reservedGradientPrefix() + i1),
            popart.TensorId(popart.reservedGradientPrefix() + o)
        ]

    def reference(ref_data):
        a = torch.tensor(data, requires_grad=True)
        ref_out = logsoftmax_reference(a, axis)

        d__o = ref_data.getOutputTensorGrad(0)
        ref_out.backward(torch.tensor(d__o))
        return [ref_out, a.grad, None]

    op_tester.atol *= 10
    op_tester.rtol *= 10
    op_tester.setPatterns([], enableRuntimeAsserts=False)
    op_tester.inplacing = inplace
    op_tester.run(init_builder, reference, 'train')
