# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import itertools
import numpy as np
import popart
import math
import torch
import pytest
from op_tester import op_tester

default_alpha = 0.01


def leaky_relu(x, alpha=default_alpha):
    return np.where(x > 0, x, x * alpha)


@pytest.mark.parametrize("inplace", [True, False])
# Using None to test default alpha op
@pytest.mark.parametrize("alpha", [None, 0.02, 1.5])
@pytest.mark.parametrize("use_torch", [True, False])
def test_lrelu_inf(op_tester, inplace, alpha, use_torch):
    input_data = np.linspace(-10, 10, 100, dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)

        if (alpha == None):
            o = builder.aiOnnx.leakyrelu([i1])
        else:
            o = builder.aiOnnx.leakyrelu([i1], alpha=alpha)

        builder.addOutputTensor(o)
        return [o]

    if (alpha == None):
        alpha = default_alpha

    def reference(ref_data):
        if (use_torch):
            torch_test_data = torch.tensor(input_data, requires_grad=False)
            torch_leakyrelu = torch.nn.LeakyReLU(negative_slope=alpha)
            m = torch_leakyrelu(torch_test_data)
        else:
            m = leaky_relu(input_data, alpha)
        return [m]

    op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    op_tester.inplacing = inplace
    op_tester.run(init_builder, reference, 'infer')


@pytest.mark.parametrize("inplace", [True, False])
# Using None to test default alpha op
@pytest.mark.parametrize("alpha", [None, 0.02, 1.5])
def test_lrelu_train(op_tester, alpha, inplace):
    input_data = np.linspace(-10, 10, 100, dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)

        if (alpha == None):
            o = builder.aiOnnx.leakyrelu([i1])
        else:
            o = builder.aiOnnx.leakyrelu([i1], alpha=alpha)

        builder.addOutputTensor(o)
        return [
            o,
            popart.TensorId(popart.reservedGradientPrefix() + i1),
            popart.TensorId(popart.reservedGradientPrefix() + o)
        ]

    if (alpha == None):
        alpha = default_alpha

    def reference(ref_data):
        a = torch.tensor(input_data, requires_grad=True)
        torch_lrelu = torch.nn.LeakyReLU(negative_slope=alpha)
        b = torch_lrelu(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    op_tester.inplacing = inplace
    op_tester.run(init_builder, reference, 'train')
