# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from enum import auto
import popxl
import popxl.ops as ops
import numpy as np
import torch
from torch import nn


class TestAveragePool:
    def test_avgpool2d(self):
        batch_size = 2
        in_channel = 10
        height = 50
        width = 50
        kernel = (3, 3)
        strides = (1, 1)
        t = np.random.rand(batch_size, in_channel, height,
                           width).astype('float32')
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream([batch_size, in_channel, height, width],
                                      popxl.float32,
                                      name="in_stream_0")
            a = ops.host_load(input0, "a")
            # custom averagepool.
            o = ops.average_pool2d(a, kernel, strides, ceil_mode=True)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        session = popxl.Session(ir, "ipu_model")
        outputs = session.run({input0: t})
        # averagepool in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_outputs = nn.AvgPool2d(kernel, stride=strides, ceil_mode=True)
        torch_output_data = torch_outputs(torch_t)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(torch_output_data.detach().numpy(),
                                   list(outputs.values())[0],
                                   rtol=1e-4,
                                   atol=1e-4)

    def test_avgpool2d_pad(self):
        batch_size = 2
        in_channel = 10
        height = 50
        width = 50
        kernel = (5, 5)
        strides = (2, 2)
        padding = (1, 1, 1, 1)
        t = np.random.rand(batch_size, in_channel, height,
                           width).astype('float32')
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream([batch_size, in_channel, height, width],
                                      popxl.float32,
                                      name="in_stream_0")
            a = ops.host_load(input0, "a")
            # custom averagepool.
            o = ops.average_pool2d(a, kernel, strides, padding, ceil_mode=True)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        session = popxl.Session(ir, "ipu_model")
        outputs = session.run({input0: t})
        # averagepool in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_outputs = nn.AvgPool2d(kernel,
                                     stride=strides,
                                     padding=(1, 1),
                                     ceil_mode=True,
                                     count_include_pad=False)
        torch_output_data = torch_outputs(torch_t)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(torch_output_data.detach().numpy(),
                                   list(outputs.values())[0],
                                   rtol=1e-4,
                                   atol=1e-4)

    def test_avgpool2d_auto_pad(self):
        batch_size = 2
        in_channel = 10
        height = 50
        width = 50
        kernel = (3, 3)
        strides = (1, 1)
        t = np.random.rand(batch_size, in_channel, height,
                           width).astype('float32')
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream([batch_size, in_channel, height, width],
                                      popxl.float32,
                                      name="in_stream_0")
            a = ops.host_load(input0, "a")
            # custom averagepool.
            o = ops.average_pool2d(a,
                                   kernel,
                                   strides,
                                   ceil_mode=True,
                                   auto_pad='VALID')
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        session = popxl.Session(ir, "ipu_model")
        outputs = session.run({input0: t})
        # averagepool in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_outputs = nn.AvgPool2d(kernel, stride=strides, ceil_mode=True)
        torch_output_data = torch_outputs(torch_t)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(torch_output_data.detach().numpy(),
                                   list(outputs.values())[0],
                                   rtol=1e-4,
                                   atol=1e-4)
