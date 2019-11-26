import itertools
import numpy as np
import popart
import math
import torch
import pytest
from op_tester import op_tester


def poplibs_gelu(x):
    return x * 0.5 * (1. + torch.tanh(7.978845608000000 * 1e-01 * x +
                                      3.567740813617200 * 1e-2 * x**3))


def test_gelu(op_tester):
    input_data = np.linspace(-10, 10, 100, dtype=np.float32)

    op_tester.atol = 1e-6

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiGraphcore.gelu([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        torch_test_data = torch.tensor(input_data, requires_grad=False)
        # torch_gelu = torch.nn.functional.gelu
        torch_gelu = poplibs_gelu
        m = torch_gelu(torch_test_data)
        return [m]

    op_tester.run(init_builder, reference, 'infer')


def test_gelu_inplace(op_tester):
    input_data = np.linspace(-10, 10, 100, dtype=np.float32)

    op_tester.atol = 1e-6

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiGraphcore.gelu([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        torch_test_data = torch.tensor(input_data, requires_grad=False)
        # torch_gelu = torch.nn.functional.gelu
        torch_gelu = poplibs_gelu
        m = torch_gelu(torch_test_data)
        return [m]

    op_tester.passes = ['InPlace']
    op_tester.run(init_builder, reference, 'infer')


def test_gelu_torch(op_tester):
    input_data = np.linspace(-10, 10, 100, dtype=np.float32)

    op_tester.atol = 1e-3

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiGraphcore.gelu([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        torch_test_data = torch.tensor(input_data, requires_grad=False)
        torch_gelu = torch.nn.functional.gelu
        m = torch_gelu(torch_test_data)
        return [m]

    op_tester.passes = ['InPlace']
    op_tester.run(init_builder, reference, 'infer')


def test_gelu_training(op_tester):
    # The regroupinBeneficial function assumes that the input tensor is a 2D one.
    # I increase the dimension of the tensor in order to avoid segfaul.
    input_data = np.asarray([np.linspace(-10, 10, 100, dtype=np.float32)])

    op_tester.atol = 1e-6

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiGraphcore.gelu([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        a = torch.tensor(input_data, requires_grad=True)
        # torch_gelu = torch.nn.functional.gelu
        torch_gelu = poplibs_gelu
        b = torch_gelu(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.passes = ['OpToIdentity']
    op_tester.run(init_builder, reference, 'train')


def test_gelu_torch_training(op_tester):
    # The regroupinBeneficial function assumes that the input tensor is a 2D one.
    # I increase the dimension of the tensor in order to avoid segfaul.
    input_data = np.asarray([np.linspace(-10, 10, 100, dtype=np.float32)])

    op_tester.atol = 1e-3

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiGraphcore.gelu([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        a = torch.tensor(input_data, requires_grad=True)
        torch_gelu = torch.nn.functional.gelu
        b = torch_gelu(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.passes = ['OpToIdentity']
    op_tester.run(init_builder, reference, 'train')
