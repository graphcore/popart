# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np
import torch
from torch import nn


class TestConv:
    def test_conv2d(self):
        batch_size = 2
        in_channel = 10
        out_channel = 20
        height = 50
        width = 50
        h_kernel = 3
        w_kernel = 3
        strides = (1, 1)
        pads = (0, 0, 0, 0)
        dilations = (1, 1)
        group = 1
        t = np.random.rand(batch_size, in_channel, height,
                           width).astype('float32')
        weight = np.random.rand(out_channel, int(in_channel / group), h_kernel,
                                w_kernel).astype('float32')
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream([batch_size, in_channel, height, width],
                                      popxl.float32,
                                      name="in_stream_0")
            a = ops.host_load(input0, "a")
            input1 = popxl.h2d_stream(
                [out_channel,
                 int(in_channel / group), h_kernel, w_kernel],
                popxl.float32,
                name="in_stream_1")
            b = ops.host_load(input1, "b")
            # custom conv.
            o = ops.conv2d(a, b, strides, pads, dilations, group, 'NOTSET')
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        session = popxl.Session(ir, "ipu_model")
        outputs = session.run({input0: t, input1: weight})
        # conv in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_weight = torch.nn.Parameter(
            torch.tensor(weight).type(torch.float32))
        torch_outputs = nn.Conv2d(in_channel,
                                  out_channel, (h_kernel, w_kernel),
                                  tuple(strides),
                                  groups=group,
                                  bias=False)
        torch_outputs.weight = torch_weight
        torch_output_data = torch_outputs(torch_t)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(torch_output_data.detach().numpy(),
                                   list(outputs.values())[0],
                                   rtol=1e-4,
                                   atol=1e-4)

    def test_conv2d_pad(self):
        batch_size = 2
        in_channel = 10
        out_channel = 20
        height = 50
        width = 50
        h_kernel = 5
        w_kernel = 5
        strides = (3, 3)
        pads = (1, 1, 1, 1)
        dilations = (1, 1)
        group = 2
        t = np.random.rand(batch_size, in_channel, height,
                           width).astype('float32')
        weight = np.random.rand(out_channel, int(in_channel / group), h_kernel,
                                w_kernel).astype('float32')
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream([batch_size, in_channel, height, width],
                                      popxl.float32,
                                      name="in_stream_0")
            a = ops.host_load(input0, "a")
            input1 = popxl.h2d_stream(
                [out_channel,
                 int(in_channel / group), h_kernel, w_kernel],
                popxl.float32,
                name="in_stream_1")
            b = ops.host_load(input1, "b")
            # custom conv.
            o = ops.conv2d(a, b, strides, pads, dilations, group, 'NOTSET')
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        session = popxl.Session(ir, "ipu_model")
        outputs = session.run({input0: t, input1: weight})
        # conv in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_weight = torch.nn.Parameter(
            torch.tensor(weight).type(torch.float32))
        torch_outputs = nn.Conv2d(in_channel,
                                  out_channel, (h_kernel, w_kernel),
                                  tuple(strides),
                                  padding=(1, 1),
                                  groups=group,
                                  bias=False)
        torch_outputs.weight = torch_weight
        torch_output_data = torch_outputs(torch_t)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(torch_output_data.detach().numpy(),
                                   list(outputs.values())[0],
                                   rtol=1e-4,
                                   atol=1e-4)

    def test_conv2d_group_dilation(self):
        batch_size = 2
        in_channel = 10
        out_channel = 20
        height = 50
        width = 50
        h_kernel = 5
        w_kernel = 5
        strides = (3, 3)
        pads = (1, 1, 1, 1)
        dilations = (2, 2)
        group = 2
        t = np.random.rand(batch_size, in_channel, height,
                           width).astype('float32')
        weight = np.random.rand(out_channel, int(in_channel / group), h_kernel,
                                w_kernel).astype('float32')
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream([batch_size, in_channel, height, width],
                                      popxl.float32,
                                      name="in_stream_0")
            a = ops.host_load(input0, "a")
            input1 = popxl.h2d_stream(
                [out_channel,
                 int(in_channel / group), h_kernel, w_kernel],
                popxl.float32,
                name="in_stream_1")
            b = ops.host_load(input1, "b")
            # custom conv.
            o = ops.conv2d(a, b, strides, pads, dilations, group, 'NOTSET')
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        session = popxl.Session(ir, "ipu_model")
        outputs = session.run({input0: t, input1: weight})
        # conv in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_weight = torch.nn.Parameter(
            torch.tensor(weight).type(torch.float32))
        torch_outputs = nn.Conv2d(in_channel,
                                  out_channel, (h_kernel, w_kernel),
                                  tuple(strides),
                                  padding=(1, 1),
                                  dilation=dilations,
                                  groups=group,
                                  bias=False)
        torch_outputs.weight = torch_weight
        torch_output_data = torch_outputs(torch_t)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(torch_output_data.detach().numpy(),
                                   list(outputs.values())[0],
                                   rtol=1e-4,
                                   atol=1e-4)

    def test_conv2d_pad_type(self):
        batch_size = 2
        in_channel = 10
        out_channel = 20
        height = 50
        width = 50
        h_kernel = 5
        w_kernel = 5
        strides = (3, 3)
        dilations = (1, 1)
        group = 2
        t = np.random.rand(batch_size, in_channel, height,
                           width).astype('float32')
        weight = np.random.rand(out_channel, int(in_channel / group), h_kernel,
                                w_kernel).astype('float32')
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream([batch_size, in_channel, height, width],
                                      popxl.float32,
                                      name="in_stream_0")
            a = ops.host_load(input0, "a")
            input1 = popxl.h2d_stream(
                [out_channel,
                 int(in_channel / group), h_kernel, w_kernel],
                popxl.float32,
                name="in_stream_1")
            b = ops.host_load(input1, "b")
            # custom conv.
            o = ops.conv2d(a,
                           b,
                           strides,
                           dilation=dilations,
                           groups=group,
                           pad_type='VALID')
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        session = popxl.Session(ir, "ipu_model")
        outputs = session.run({input0: t, input1: weight})
        # conv in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_weight = torch.nn.Parameter(
            torch.tensor(weight).type(torch.float32))
        torch_outputs = nn.Conv2d(in_channel,
                                  out_channel, (h_kernel, w_kernel),
                                  tuple(strides),
                                  dilation=dilations,
                                  groups=group,
                                  bias=False)
        torch_outputs.weight = torch_weight
        torch_output_data = torch_outputs(torch_t)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(torch_output_data.detach().numpy(),
                                   list(outputs.values())[0],
                                   rtol=1e-4,
                                   atol=1e-4)
