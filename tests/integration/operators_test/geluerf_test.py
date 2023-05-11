# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch


def test_geluerf(op_tester):
    input_data = np.linspace(-10, 10, 100, dtype=np.float32)

    op_tester.atol = 1e-6

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiGraphcore.geluerf([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        torch_test_data = torch.tensor(input_data, requires_grad=False)
        torch_gelu = torch.nn.functional.gelu
        m = torch_gelu(torch_test_data)
        return [m]

    op_tester.run(init_builder, reference, "infer")


def test_geluerf_inplace(op_tester):
    input_data = np.linspace(-10, 10, 100, dtype=np.float32)

    op_tester.atol = 1e-6

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiGraphcore.geluerf([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        torch_test_data = torch.tensor(input_data, requires_grad=False)
        torch_gelu = torch.nn.functional.gelu
        m = torch_gelu(torch_test_data)
        return [m]

    op_tester.setPatterns(["InPlace"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "infer")


def test_geluerf_training(op_tester):
    # The regroupinBeneficial function assumes that the input tensor is a 2D one.
    # I increase the dimension of the tensor in order to avoid segfaul.
    input_data = np.asarray([np.linspace(-10, 10, 100, dtype=np.float32)])

    op_tester.atol = 1e-5

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiGraphcore.geluerf([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        a = torch.tensor(input_data, requires_grad=True)
        torch_gelu = torch.nn.functional.gelu
        b = torch_gelu(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.setPatterns(["OpToIdentity"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")
