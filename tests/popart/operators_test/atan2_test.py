# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
from op_tester import op_tester


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
