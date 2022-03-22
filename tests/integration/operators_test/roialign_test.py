# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
from torchvision.ops import RoIAlign as torch_roialign


def test_roialign_fp32(op_tester):
    # create test data
    batch_size = 2
    channel = 4
    width = 50
    height = 50
    num_roi = 256
    input_data = np.random.uniform(
        0, 2, (batch_size, channel, width, height)).astype('float32')
    input_roi = np.random.uniform(0, 100, (num_roi, 4)).astype('float32')
    input_batch_index = np.random.randint(0,
                                          batch_size, (num_roi),
                                          dtype=np.int32)

    # test parameters
    pool_height = 6
    pool_width = 6
    spatial_scale = 0.05
    sampling_ratio = 2

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        i2 = builder.addInputTensor(input_roi)
        i3 = builder.addInputTensor(input_batch_index)
        o = builder.aiOnnx.roialign([i1, i2, i3],
                                    sampling_ratio=sampling_ratio,
                                    spatial_scale=spatial_scale,
                                    output_height=pool_height,
                                    output_width=pool_width,
                                    debugContext='roialign')
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        torch_input_1 = torch.tensor(input_data, requires_grad=True)
        torch_rois = np.concatenate(
            (input_batch_index.reshape(-1, 1), input_roi), axis=1)
        torch_input_2 = torch.tensor(torch_rois).type(torch.float32)
        alignNet = torch_roialign(output_size=(pool_height, pool_width),
                                  spatial_scale=spatial_scale,
                                  sampling_ratio=sampling_ratio)
        out = alignNet(torch_input_1, torch_input_2)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, torch_input_1.grad, d__o]

    op_tester.atol = 1e-3
    op_tester.rtol = 1e-3
    op_tester.run(init_builder, reference, 'train')


def test_roialign_fp16(op_tester):
    # create test data
    batch_size = 2
    channel = 4
    width = 50
    height = 50
    num_roi = 256
    input_data = np.random.uniform(
        0, 2, (batch_size, channel, width, height)).astype('float16')
    input_roi = np.random.uniform(0, 100, (num_roi, 4)).astype('float16')
    input_batch_index = np.random.randint(0,
                                          batch_size, (num_roi),
                                          dtype=np.int32)

    # test parameters
    pool_height = 6
    pool_width = 6
    spatial_scale = 0.05
    sampling_ratio = 2

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        i2 = builder.addInputTensor(input_roi)
        i3 = builder.addInputTensor(input_batch_index)
        o = builder.aiOnnx.roialign([i1, i2, i3],
                                    sampling_ratio=sampling_ratio,
                                    spatial_scale=spatial_scale,
                                    output_height=pool_height,
                                    output_width=pool_width,
                                    debugContext='roialign')
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        torch_input_1 = torch.tensor(input_data, requires_grad=True)
        torch_rois = np.concatenate(
            (input_batch_index.reshape(-1, 1), input_roi), axis=1)
        torch_input_2 = torch.tensor(torch_rois).type(torch.float16)
        alignNet = torch_roialign(output_size=(pool_height, pool_width),
                                  spatial_scale=spatial_scale,
                                  sampling_ratio=sampling_ratio)
        out = alignNet(torch_input_1, torch_input_2)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, torch_input_1.grad, d__o]

    op_tester.atol = 1e-2
    op_tester.rtol = 2e-2
    op_tester.run(init_builder, reference, 'train')
