# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch


def test_atan2_basic(op_tester):
    vals = np.linspace(-5, 5, 21, dtype=np.float32)

    input_y, input_x = np.meshgrid(vals, vals)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_y)
        i2 = builder.addInputTensor(input_x)
        o = builder.aiGraphcore.atan2([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        torch_input_y = torch.tensor(input_y, requires_grad=False)
        torch_input_x = torch.tensor(input_x, requires_grad=False)
        out = torch.atan2(torch_input_y, torch_input_x)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_atan2_float16(op_tester):
    vals = np.linspace(-5, 5, 21, dtype=np.float16)

    input_y, input_x = np.meshgrid(vals, vals)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_y)
        i2 = builder.addInputTensor(input_x)
        o = builder.aiGraphcore.atan2([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        torch_input_y = torch.tensor(input_y, requires_grad=False)
        torch_input_x = torch.tensor(input_x, requires_grad=False)

        # 16 bit-not supported on cpu
        out = torch.atan2(torch_input_y.to(torch.float32),
                          torch_input_x.to(torch.float32)).to(torch.float16)
        return [out]

    op_tester.atol = 1e-02  #Allow for 16 vs 32 bit
    op_tester.run(init_builder, reference, 'infer')


def test_atan2_reshaped(op_tester):
    vals = np.linspace(-5, 5, 21, dtype=np.float32)

    input_y, input_x = np.meshgrid(vals, vals)
    input_y = np.reshape(input_y, [7, 3, -1])
    input_x = np.reshape(input_x, [7, 3, -1])

    def init_builder(builder):
        i1 = builder.addInputTensor(input_y)
        i2 = builder.addInputTensor(input_x)
        o = builder.aiGraphcore.atan2([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        torch_input_y = torch.tensor(input_y, requires_grad=False)
        torch_input_x = torch.tensor(input_x, requires_grad=False)
        out = torch.atan2(torch_input_y, torch_input_x)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_atan2_broadcasting_y(op_tester):
    input_x = np.array([0.0], dtype=np.float32)
    input_y = np.linspace(-5, 5, 21, dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_y)
        i2 = builder.addInputTensor(input_x)
        o = builder.aiGraphcore.atan2([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        torch_input_y = torch.tensor(input_y, requires_grad=False)
        torch_input_x = torch.tensor(input_x, requires_grad=False)
        out = torch.atan2(torch_input_y, torch_input_x)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_atan2_outplace(op_tester):
    vals = np.linspace(-5, 5, 21, dtype=np.float32)

    input_y, input_x = np.meshgrid(vals, vals)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_y)
        i2 = builder.addInputTensor(input_x)
        o = builder.aiGraphcore.atan2([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        torch_input_y = torch.tensor(input_y, requires_grad=False)
        torch_input_x = torch.tensor(input_x, requires_grad=False)
        out = torch.atan2(torch_input_y, torch_input_x)
        return [out]

    op_tester.inplacing = False
    op_tester.run(init_builder, reference, 'infer')


def test_atan2_grad(op_tester):
    vals = np.linspace(-5, 5, 21, dtype=np.float32)
    input_y, input_x = np.meshgrid(vals, vals)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_y)
        i2 = builder.addInputTensor(input_x)
        o = builder.aiGraphcore.atan2([i1, i2])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        torch_input_y = torch.tensor(input_y, requires_grad=True)
        torch_input_x = torch.tensor(input_x, requires_grad=True)
        out = torch.atan2(torch_input_y, torch_input_x)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, torch_input_y.grad, torch_input_x.grad, None]

    # Need to have NaN == NaN to mirror numpy's functionality
    op_tester.equal_nan = True
    op_tester.run(init_builder, reference, 'train')
