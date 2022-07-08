# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch


def test_hardshrink(op_tester):
    input_data = np.linspace(-10, 10, 250, dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiOnnx.shrink([i1], lambd=1.5)
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        torch_test_data = torch.tensor(input_data, requires_grad=False)
        torch_hardshrink = torch.nn.Hardshrink(lambd=1.5)
        m = torch_hardshrink(torch_test_data)
        return [m]

    op_tester.run(init_builder, reference, "infer")


def test_softshrink(op_tester):
    input_data = np.linspace(-10, 10, 250, dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiOnnx.shrink([i1], lambd=1.5, bias=1.5)
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        torch_test_data = torch.tensor(input_data, requires_grad=False)
        torch_softshrink = torch.nn.Softshrink(lambd=1.5)
        m = torch_softshrink(torch_test_data)
        return [m]

    op_tester.run(init_builder, reference, "infer")


def test_hardshrink_training(op_tester):
    input_data = np.linspace(-10, 10, 250, dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiOnnx.shrink([i1], lambd=1.5)
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        a = torch.tensor(input_data, requires_grad=True)
        hs = torch.nn.Hardshrink(lambd=1.5)
        b = hs(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.setPatterns(["OpToIdentity"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")


def test_softshrink_training(op_tester):
    input_data = np.linspace(-10, 10, 250, dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiOnnx.shrink([i1], lambd=1.5, bias=1.5)
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        a = torch.tensor(input_data, requires_grad=True)
        hs = torch.nn.Softshrink(lambd=1.5)
        b = hs(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.setPatterns(["OpToIdentity"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")
