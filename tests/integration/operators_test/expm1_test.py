# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch


def test_expm1_0(op_tester):
    input_data = np.array([-2, -1, 0, 1, 2], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiGraphcore.expm1([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        torch_test_data = torch.tensor(input_data, requires_grad=False)
        out = torch.add(torch.exp(torch_test_data), -1.0)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_expm1_1(op_tester):
    input_data = np.linspace(-10, 10, 100, dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiGraphcore.expm1([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        torch_test_data = torch.tensor(input_data, requires_grad=False)
        out = torch.add(torch.exp(torch_test_data), -1.0)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_expm1_2(op_tester):
    input_data = np.linspace(-10, 10, 100, dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiGraphcore.expm1([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        torch_test_data = torch.tensor(input_data, requires_grad=False)
        out = torch.expm1(torch_test_data)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


# Similar to gelu_test.py
def test_expm1_inplace_0(op_tester):
    input_data = np.linspace(-10, 10, 100, dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiGraphcore.expm1([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        torch_test_data = torch.tensor(input_data, requires_grad=False)
        out = torch.add(torch.exp(torch_test_data), -1.0)
        return [out]

    op_tester.setPatterns(['InPlace'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


# Similar to exp_test.py
def test_expm1_inplace_1(op_tester):
    """
    Test of both Expm1Inplace (priority > 0) and Expm1 (priority <= 0)
    """
    d2 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    d3 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    for inplace_priority in [-100., +100.]:
        d1 = np.random.rand(4).astype(np.float32)

        def init_builder(builder):
            i1 = builder.addInputTensor(d1)
            i2 = builder.addInputTensor(d2)
            i3 = builder.addInputTensor(d3)
            e0 = builder.aiGraphcore.expm1([i1])
            o0 = builder.aiOnnx.add([e0, i2])
            builder.setInplacePreferences(e0,
                                          {"Expm1Inplace": inplace_priority})
            o1 = builder.aiOnnx.log([o0])
            e1 = builder.aiGraphcore.expm1([o1])
            o2 = builder.aiOnnx.add([e1, i3])
            builder.setInplacePreferences(e1,
                                          {"Expm1Inplace": inplace_priority})
            o3 = builder.aiOnnx.log([o2])
            builder.addOutputTensor(o3)
            return [o3]

        def reference(_):  # ref_data is an unused argument
            """
            add1(expm1(log(add1(expm1(log))))) (x) = x
            """
            a = torch.tensor(d1, requires_grad=True)
            return [a * 1.0]

        op_tester.run(init_builder, reference, 'infer')


def test_expm1_inplace_2(op_tester):
    """
    Test of both
    1) 1 Expm1Inplace and 2 Expm1 (priority > 0) and
    2) 3 Expm1 (priority <= 0)
    """
    for inplace_priority in [-100., +100.]:
        d1 = np.random.rand(4).astype(np.float32)

        def init_builder(builder):
            i1 = builder.addInputTensor(d1)
            # expm1 1
            o1 = builder.aiGraphcore.expm1([i1])
            builder.setInplacePreferences(o1,
                                          {"Expm1Inplace": inplace_priority})
            # expm1 2
            o2 = builder.aiGraphcore.expm1([i1])
            builder.setInplacePreferences(o2,
                                          {"Expm1Inplace": inplace_priority})
            # expm1 3
            o3 = builder.aiGraphcore.expm1([i1])
            builder.setInplacePreferences(o3,
                                          {"Expm1Inplace": inplace_priority})

            o4 = builder.aiOnnx.sum([o1, o2, o3])
            return [o4]

        def reference(_):  # ref_data is an unused argument
            """
            3*expm1(in)
            """
            a = torch.tensor(3 * np.add(np.exp(d1), -1.0), requires_grad=True)
            return [a]

        op_tester.run(init_builder, reference, 'infer')


def test_expm1_grad_0(op_tester):
    d1 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiGraphcore.expm1([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.expm1(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.setPatterns(['PreUniRepl', 'Expm1GradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_expm1_grad_1(op_tester):
    d1 = np.linspace(-10, 10, 100, dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiGraphcore.expm1([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.expm1(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.setPatterns(['PreUniRepl', 'Expm1GradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')
