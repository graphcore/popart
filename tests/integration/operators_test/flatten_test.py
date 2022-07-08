# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import torch
import popart


def test_flatten(op_tester):
    d1 = np.random.rand(2, 1, 3, 4, 1).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.flatten([i1], axis=0)
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        o = d1.flatten().reshape(1, -1)
        return [o]

    op_tester.setPatterns([], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "infer")


def test_flatten_negative(op_tester):
    d1 = np.random.rand(2, 1, 3, 4, 5).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.flatten([i1], axis=-2)
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        o = d1.reshape(6, 20)
        return [o]

    op_tester.setPatterns([], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "infer")


def test_flatten_grad(op_tester):
    d1 = np.random.rand(2, 1, 3, 4, 1).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.flatten([i1], axis=0)
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        i1 = torch.tensor(d1, requires_grad=True)
        o = torch.flatten(i1)
        o = torch.reshape(o, (1, -1))
        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))
        return [o, i1.grad, None]

    op_tester.setPatterns(["PreUniRepl"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")
