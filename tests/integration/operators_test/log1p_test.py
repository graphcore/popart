# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch


def test_log1p_0(op_tester):
    input_data = np.array([0.01, 0.1, 0.5, 1, 2, 12345], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiGraphcore.log1p([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        torch_test_data = torch.tensor(input_data, requires_grad=False)
        xp1 = torch.add(torch_test_data, 1.0)
        out = torch.log(xp1)
        return [out]

    op_tester.run(init_builder, reference, "infer")


def test_log1p_1(op_tester):
    input_data = np.linspace(0, 1e9, 100, dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiGraphcore.log1p([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        torch_test_data = torch.tensor(input_data, requires_grad=False)
        out = torch.log1p(torch_test_data)
        return [out]

    op_tester.run(init_builder, reference, "infer")


def test_log1p_nan(op_tester):
    input_data = np.array([-10000.1, -123.1, -2.0, -1.55, -1.0, 10.0], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiGraphcore.log1p([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        torch_test_data = torch.tensor(input_data, requires_grad=False)
        out = torch.log1p(torch_test_data)
        return [out]

    op_tester.equal_nan = True
    op_tester.run(init_builder, reference, "infer")


# Similar to gelu_test.py
def test_log1p_inplace_0(op_tester):
    input_data = np.array(
        [-0.99, -0.5, 0.01, 0.1, 0.333, 0.5, 0.789, 1, 2, 8.8, 1212.1, 12345.6],
        dtype=np.float32,
    )

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiGraphcore.log1p([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        torch_test_data = torch.tensor(input_data, requires_grad=False)
        out = torch.log1p(torch_test_data)
        return [out]

    op_tester.setPatterns(["InPlace"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "infer")


# Similar to exp_test.py
def test_log1p_inplace_1(op_tester):
    """
    Test of both Log1pInplace (priority > 0) and Log1p (priority <= 0)
    """
    d2 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    d3 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    for inplace_priority in [-100.0, +100.0]:
        d1 = np.random.rand(4).astype(np.float32)

        def init_builder(builder):
            i1 = builder.addInputTensor(d1)
            i2 = builder.addInputTensor(d2)
            i3 = builder.addInputTensor(d3)
            a1 = builder.aiOnnx.sub([i1, i2])
            o0 = builder.aiGraphcore.log1p([a1])
            builder.setInplacePreferences(o0, {"Log1pInplace": inplace_priority})
            o1 = builder.aiOnnx.exp([o0])
            a2 = builder.aiOnnx.sub([o1, i3])
            o2 = builder.aiGraphcore.log1p([a2])
            builder.setInplacePreferences(o2, {"Log1pInplace": inplace_priority})
            o3 = builder.aiOnnx.exp([o2])
            builder.addOutputTensor(o3)
            return [o3]

        def reference(_):  # ref_data is an unused argument
            """
            exp(logp1(sub(exp(logp1(sub))))(x) = x
            """
            a = torch.tensor(d1, requires_grad=True)
            return [a * 1.0]

        op_tester.run(init_builder, reference, "infer")


def test_log1p_inplace_2(op_tester):
    """
    Test of both
    1) 1 Log1pInplace and 2 Log1p (priority > 0) and
    2) 3 Log1p (priority <= 0)
    """
    for inplace_priority in [-100.0, +100.0]:
        d1 = np.random.rand(4).astype(np.float32)

        def init_builder(builder):
            i1 = builder.addInputTensor(d1)
            # log1p 1
            o1 = builder.aiGraphcore.log1p([i1])
            builder.setInplacePreferences(o1, {"Log1pInplace": inplace_priority})
            # log1p 2
            o2 = builder.aiGraphcore.log1p([i1])
            builder.setInplacePreferences(o2, {"Log1pInplace": inplace_priority})
            # log1p 3
            o3 = builder.aiGraphcore.log1p([i1])
            builder.setInplacePreferences(o3, {"Log1pInplace": inplace_priority})

            o4 = builder.aiOnnx.sum([o1, o2, o3])
            return [o4]

        def reference(_):  # ref_data is an unused argument
            """
            3*log1p(in)
            """
            a = torch.tensor(3 * np.log1p(d1), requires_grad=True)
            return [a]

        op_tester.run(init_builder, reference, "infer")


def test_log1p_grad_0(op_tester):
    d1 = np.array(
        [-0.99, -0.5, 0.01, 0.1, 0.333, 0.5, 0.789, 1, 2, 8.8, 1212.1, 12345.6],
        dtype=np.float32,
    )

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiGraphcore.log1p([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.log1p(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.setPatterns(["PreUniRepl", "Log1pGradOp"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")


def test_log1p_grad_1(op_tester):
    d1 = np.linspace(0, 1e9, 100, dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiGraphcore.log1p([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.log1p(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.setPatterns(["PreUniRepl", "Log1pGradOp"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")


def test_log1p_grad_nan(op_tester):
    d1 = np.array([-10000.1, -123.1, -2.0, -1.55, -1.0, 10.0], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiGraphcore.log1p([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.log1p(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.setPatterns(["PreUniRepl", "Log1pGradOp"], enableRuntimeAsserts=False)
    op_tester.equal_nan = True
    op_tester.run(init_builder, reference, "train")
