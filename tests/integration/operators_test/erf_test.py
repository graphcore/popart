# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
import pytest
from op_tester import op_tester


def test_erf_0(op_tester):
    x = np.array([0., -1., 10.]).astype(np.float32)

    def init_builder(builder):
        i0 = builder.addInputTensor(x)
        o = builder.aiOnnx.erf([i0])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        tx = torch.tensor(x)
        out = torch.erf(tx)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_erf_0b(op_tester, dtype):
    x = np.array([0., -1., 10.]).astype(dtype)
    expected = np.array([0., -0.84270079, 1.]).astype(dtype)

    def init_builder(builder):
        i0 = builder.addInputTensor(x)
        o = builder.aiOnnx.erf([i0])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        return [expected]

    # Lower precision for float16
    if dtype == np.float16:
        op_tester.atol = 1e-03
    op_tester.run(init_builder, reference, 'infer')


def test_erf_1(op_tester):
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)

    def init_builder(builder):
        i0 = builder.addInputTensor(x)
        o = builder.aiOnnx.erf([i0])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        tx = torch.tensor(x)
        out = torch.erf(tx)
        return [out]

    op_tester.atol = 1e-05
    op_tester.run(init_builder, reference, 'infer')


def test_erf_2(op_tester):
    x = np.random.uniform(low=-10., high=20.3,
                          size=(1, 2, 3, 20)).astype(np.float32)

    def init_builder(builder):
        i0 = builder.addInputTensor(x)
        o = builder.aiOnnx.erf([i0])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        tx = torch.tensor(x)
        out = torch.erf(tx)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_erf_grad_0(op_tester):
    x = np.array([0., -1., 10.]).astype(np.float32)

    def init_builder(builder):
        i0 = builder.addInputTensor(x)
        o = builder.aiOnnx.erf([i0])
        builder.addOutputTensor(o)
        return [
            o,
            popart.TensorId(popart.reservedGradientPrefix() + i0),
            popart.TensorId(popart.reservedGradientPrefix() + o),
        ]

    def reference(ref_data):
        tx = torch.tensor(x, requires_grad=True)
        out = torch.erf(tx)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, None]

    op_tester.run(init_builder, reference, 'train')


def test_erf_grad_0b(op_tester):
    # dx erf(x) = 2/Sqrt(Pi) exp(-x^2)
    # pytorch does not support erf() for float16
    def derf(x):
        return (2.0 / np.sqrt(np.pi)) * np.exp(-np.square(x))

    tesTypes = [np.float16, np.float32]

    for tesType in tesTypes:
        x = np.array([0., -1., 10.]).astype(tesType)
        expectedErf = np.array([0., -0.84270079, 1.]).astype(tesType)

        def init_builder(builder):
            i0 = builder.addInputTensor(x)
            o = builder.aiOnnx.erf([i0])
            builder.addOutputTensor(o)
            return [
                o,
                popart.TensorId(popart.reservedGradientPrefix() + i0),
                popart.TensorId(popart.reservedGradientPrefix() + o),
            ]

        def reference(ref_data):
            out = expectedErf
            d__o = derf(x) * ref_data.getOutputTensorGrad(0)
            return [out, d__o, None]

        # Lower precision for float16
        op_tester.atol = 1e-03
        op_tester.run(init_builder, reference, 'train')


def test_erf_grad_1(op_tester):
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)

    def init_builder(builder):
        i0 = builder.addInputTensor(x)
        o = builder.aiOnnx.erf([i0])
        builder.addOutputTensor(o)
        return [
            o,
            popart.TensorId(popart.reservedGradientPrefix() + i0),
            popart.TensorId(popart.reservedGradientPrefix() + o),
        ]

    def reference(ref_data):
        tx = torch.tensor(x, requires_grad=True)
        out = torch.erf(tx)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, None]

    op_tester.atol = 1e-05
    op_tester.run(init_builder, reference, 'train')


def test_erf_grad_2(op_tester):
    x = np.random.uniform(low=-10., high=20.3,
                          size=(1, 2, 3, 20)).astype(np.float32)

    def init_builder(builder):
        i0 = builder.addInputTensor(x)
        o = builder.aiOnnx.erf([i0])
        builder.addOutputTensor(o)
        return [
            o,
            popart.TensorId(popart.reservedGradientPrefix() + i0),
            popart.TensorId(popart.reservedGradientPrefix() + o),
        ]

    def reference(ref_data):
        tx = torch.tensor(x, requires_grad=True)
        out = torch.erf(tx)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, None]

    op_tester.run(init_builder, reference, 'train')
