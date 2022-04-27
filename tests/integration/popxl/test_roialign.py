# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl
import popxl.ops as ops
import torch
from torchvision.ops import RoIAlign as torch_roialign


class TestRoiAlign:
    def test_roi_align(self):
        output_size = (6, 6)
        spatial_scale = 0.05
        sampling_ratio = 2
        batch_size = 1
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
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream([batch_size, channel, height, width],
                                      popxl.float32,
                                      name="in_stream_0")
            a = ops.host_load(input0, "a")
            input1 = popxl.h2d_stream([num_roi, 4],
                                      popxl.float32,
                                      name="in_stream_1")
            b = ops.host_load(input1, "b")
            input2 = popxl.h2d_stream([num_roi],
                                      popxl.int32,
                                      name="in_stream_2")
            c = ops.host_load(input2, "c")
            # custom roialign.
            o = ops.roi_align(a, b, c, output_size, spatial_scale,
                              sampling_ratio)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({
                input0: input_data,
                input1: input_roi,
                input2: input_batch_index
            })
        # roialign in torch
        torch_input_data = torch.tensor(input_data)
        torch_rois = np.concatenate(
            (input_batch_index.reshape(-1, 1), input_roi), axis=1)
        torch_input_roi = torch.tensor(torch_rois).type(torch.float32)
        alignNet = torch_roialign(output_size=output_size,
                                  spatial_scale=spatial_scale,
                                  sampling_ratio=sampling_ratio)
        torch_output_data = alignNet(torch_input_data, torch_input_roi)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(torch_output_data.detach().numpy(),
                                   list(outputs.values())[0],
                                   rtol=1e-4,
                                   atol=1e-4)

    def test_roi_align_batch(self):
        output_size = (12, 12)
        spatial_scale = 0.5
        sampling_ratio = 3
        batch_size = 5
        channel = 4
        width = 25
        height = 25
        num_roi = 128
        input_data = np.random.uniform(
            0, 2, (batch_size, channel, width, height)).astype('float32')
        input_roi = np.random.uniform(0, 100, (num_roi, 4)).astype('float32')
        input_batch_index = np.random.randint(0,
                                              batch_size, (num_roi),
                                              dtype=np.int32)
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream([batch_size, channel, height, width],
                                      popxl.float32,
                                      name="in_stream_0")
            a = ops.host_load(input0, "a")
            input1 = popxl.h2d_stream([num_roi, 4],
                                      popxl.float32,
                                      name="in_stream_1")
            b = ops.host_load(input1, "b")
            input2 = popxl.h2d_stream([num_roi],
                                      popxl.int32,
                                      name="in_stream_2")
            c = ops.host_load(input2, "c")
            # custom roialign.
            o = ops.roi_align(a, b, c, output_size, spatial_scale,
                              sampling_ratio)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({
                input0: input_data,
                input1: input_roi,
                input2: input_batch_index
            })
        # roialign in torch
        torch_input_data = torch.tensor(input_data)
        torch_rois = np.concatenate(
            (input_batch_index.reshape(-1, 1), input_roi), axis=1)
        torch_input_roi = torch.tensor(torch_rois).type(torch.float32)
        alignNet = torch_roialign(output_size=output_size,
                                  spatial_scale=spatial_scale,
                                  sampling_ratio=sampling_ratio)
        torch_output_data = alignNet(torch_input_data, torch_input_roi)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(torch_output_data.detach().numpy(),
                                   list(outputs.values())[0],
                                   rtol=1e-4,
                                   atol=1e-4)
