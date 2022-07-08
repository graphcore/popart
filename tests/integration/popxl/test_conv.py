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
        group = 1
        t = np.random.rand(batch_size, in_channel, height, width).astype("float32")
        weight = np.random.rand(
            out_channel, int(in_channel / group), h_kernel, w_kernel
        ).astype("float32")
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream(
                [batch_size, in_channel, height, width],
                popxl.float32,
                name="in_stream_0",
            )
            a = ops.host_load(input0, "a")
            input1 = popxl.h2d_stream(
                [out_channel, int(in_channel / group), h_kernel, w_kernel],
                popxl.float32,
                name="in_stream_1",
            )
            b = ops.host_load(input1, "b")
            # custom conv.
            o = ops.conv(a, b, strides)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: t, input1: weight})
        # conv in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_weight = torch.nn.Parameter(torch.tensor(weight).type(torch.float32))
        torch_outputs = nn.Conv2d(
            in_channel,
            out_channel,
            (h_kernel, w_kernel),
            tuple(strides),
            groups=group,
            bias=False,
        )
        torch_outputs.weight = torch_weight
        torch_output_data = torch_outputs(torch_t)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(
            torch_output_data.detach().numpy(),
            list(outputs.values())[0],
            rtol=1e-5,
            atol=1e-5,
        )

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
        t = np.random.rand(batch_size, in_channel, height, width).astype("float32")
        weight = np.random.rand(
            out_channel, int(in_channel / group), h_kernel, w_kernel
        ).astype("float32")
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream(
                [batch_size, in_channel, height, width],
                popxl.float32,
                name="in_stream_0",
            )
            a = ops.host_load(input0, "a")
            input1 = popxl.h2d_stream(
                [out_channel, int(in_channel / group), h_kernel, w_kernel],
                popxl.float32,
                name="in_stream_1",
            )
            b = ops.host_load(input1, "b")
            # custom conv.
            o = ops.conv(a, b, strides, pads, dilations, group)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: t, input1: weight})
        # conv in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_weight = torch.nn.Parameter(torch.tensor(weight).type(torch.float32))
        torch_outputs = nn.Conv2d(
            in_channel,
            out_channel,
            (h_kernel, w_kernel),
            tuple(strides),
            padding=(1, 1),
            groups=group,
            bias=False,
        )
        torch_outputs.weight = torch_weight
        torch_output_data = torch_outputs(torch_t)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(
            torch_output_data.detach().numpy(),
            list(outputs.values())[0],
            rtol=1e-5,
            atol=1e-5,
        )

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
        t = np.random.rand(batch_size, in_channel, height, width).astype("float32")
        weight = np.random.rand(
            out_channel, int(in_channel / group), h_kernel, w_kernel
        ).astype("float32")
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream(
                [batch_size, in_channel, height, width],
                popxl.float32,
                name="in_stream_0",
            )
            a = ops.host_load(input0, "a")
            input1 = popxl.h2d_stream(
                [out_channel, int(in_channel / group), h_kernel, w_kernel],
                popxl.float32,
                name="in_stream_1",
            )
            b = ops.host_load(input1, "b")
            # custom conv.
            o = ops.conv(a, b, strides, pads, dilations, group, "not_set")
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: t, input1: weight})
        # conv in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_weight = torch.nn.Parameter(torch.tensor(weight).type(torch.float32))
        torch_outputs = nn.Conv2d(
            in_channel,
            out_channel,
            (h_kernel, w_kernel),
            tuple(strides),
            padding=(1, 1),
            dilation=dilations,
            groups=group,
            bias=False,
        )
        torch_outputs.weight = torch_weight
        torch_output_data = torch_outputs(torch_t)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(
            torch_output_data.detach().numpy(),
            list(outputs.values())[0],
            rtol=1e-5,
            atol=1e-5,
        )

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
        t = np.random.rand(batch_size, in_channel, height, width).astype("float32")
        weight = np.random.rand(
            out_channel, int(in_channel / group), h_kernel, w_kernel
        ).astype("float32")
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream(
                [batch_size, in_channel, height, width],
                popxl.float32,
                name="in_stream_0",
            )
            a = ops.host_load(input0, "a")
            input1 = popxl.h2d_stream(
                [out_channel, int(in_channel / group), h_kernel, w_kernel],
                popxl.float32,
                name="in_stream_1",
            )
            b = ops.host_load(input1, "b")
            # custom conv.
            o = ops.conv(
                a, b, strides, dilation=dilations, groups=group, pad_type="valid"
            )
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: t, input1: weight})
        # conv in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_weight = torch.nn.Parameter(torch.tensor(weight).type(torch.float32))
        torch_outputs = nn.Conv2d(
            in_channel,
            out_channel,
            (h_kernel, w_kernel),
            tuple(strides),
            dilation=dilations,
            groups=group,
            bias=False,
        )
        torch_outputs.weight = torch_weight
        torch_output_data = torch_outputs(torch_t)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(
            torch_output_data.detach().numpy(),
            list(outputs.values())[0],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_conv1d(self):
        batch_size = 2
        in_channel = 10
        out_channel = 20
        Length = 50
        kernel = 3
        strides = (1,)
        group = 1
        padding = (0, 0)
        dilation = (1,)
        t = np.random.rand(batch_size, in_channel, Length).astype("float32")
        weight = np.random.rand(out_channel, int(in_channel / group), kernel).astype(
            "float32"
        )
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream(
                [batch_size, in_channel, Length], popxl.float32, name="in_stream_0"
            )
            a = ops.host_load(input0, "a")
            input1 = popxl.h2d_stream(
                [out_channel, int(in_channel / group), kernel],
                popxl.float32,
                name="in_stream_1",
            )
            b = ops.host_load(input1, "b")
            # custom conv.
            o = ops.conv(a, b, strides, padding, dilation)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: t, input1: weight})
        # conv in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_weight = torch.nn.Parameter(torch.tensor(weight).type(torch.float32))
        torch_outputs = nn.Conv1d(
            in_channel, out_channel, (kernel,), strides, groups=group, bias=False
        )
        torch_outputs.weight = torch_weight
        torch_output_data = torch_outputs(torch_t)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(
            torch_output_data.detach().numpy(),
            list(outputs.values())[0],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_conv1d_pad_group_dilation(self):
        batch_size = 2
        in_channel = 10
        out_channel = 20
        Length = 50
        kernel = 3
        strides = (1,)
        group = 2
        padding = (1, 1)
        dilation = (2,)
        t = np.random.rand(batch_size, in_channel, Length).astype("float32")
        weight = np.random.rand(out_channel, int(in_channel / group), kernel).astype(
            "float32"
        )
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream(
                [batch_size, in_channel, Length], popxl.float32, name="in_stream_0"
            )
            a = ops.host_load(input0, "a")
            input1 = popxl.h2d_stream(
                [out_channel, int(in_channel / group), kernel],
                popxl.float32,
                name="in_stream_1",
            )
            b = ops.host_load(input1, "b")
            # custom conv.
            o = ops.conv(a, b, strides, padding, dilation, group)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: t, input1: weight})
        # conv in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_weight = torch.nn.Parameter(torch.tensor(weight).type(torch.float32))
        torch_outputs = nn.Conv1d(
            in_channel,
            out_channel,
            (kernel,),
            strides,
            groups=group,
            padding=1,
            dilation=dilation,
            bias=False,
        )
        torch_outputs.weight = torch_weight
        torch_output_data = torch_outputs(torch_t)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(
            torch_output_data.detach().numpy(),
            list(outputs.values())[0],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_conv3d(self):
        batch_size = 2
        in_channel = 10
        in_deepth = 10
        out_channel = 20
        height = 25
        width = 25
        d_kernel = 3
        h_kernel = 3
        w_kernel = 3
        strides = (1, 1, 1)
        group = 1
        padding = (0, 0, 0, 0, 0, 0)
        dilation = (1, 1, 1)
        t = np.random.rand(batch_size, in_channel, in_deepth, height, width).astype(
            "float32"
        )
        weight = np.random.rand(
            out_channel, int(in_channel / group), d_kernel, h_kernel, w_kernel
        ).astype("float32")
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream(
                [batch_size, in_channel, in_deepth, height, width],
                popxl.float32,
                name="in_stream_0",
            )
            a = ops.host_load(input0, "a")
            input1 = popxl.h2d_stream(
                [out_channel, int(in_channel / group), d_kernel, h_kernel, w_kernel],
                popxl.float32,
                name="in_stream_1",
            )
            b = ops.host_load(input1, "b")
            # custom conv.
            o = ops.conv(a, b, strides, padding, dilation, group)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: t, input1: weight})
        # conv in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_weight = torch.nn.Parameter(torch.tensor(weight).type(torch.float32))
        torch_outputs = nn.Conv3d(
            in_channel,
            out_channel,
            (d_kernel, h_kernel, w_kernel),
            strides,
            groups=group,
            bias=False,
        )
        torch_outputs.weight = torch_weight
        torch_output_data = torch_outputs(torch_t)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(
            torch_output_data.detach().numpy(),
            list(outputs.values())[0],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_conv3d_pad_group_dilation(self):
        batch_size = 2
        in_channel = 20
        in_deepth = 10
        out_channel = 20
        height = 25
        width = 25
        d_kernel = 2
        h_kernel = 3
        w_kernel = 4
        strides = (2, 2, 2)
        group = 2
        padding = (1, 1, 1, 1, 1, 1)
        dilation = (2, 2, 2)
        t = np.random.rand(batch_size, in_channel, in_deepth, height, width).astype(
            "float32"
        )
        weight = np.random.rand(
            out_channel, int(in_channel / group), d_kernel, h_kernel, w_kernel
        ).astype("float32")
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream(
                [batch_size, in_channel, in_deepth, height, width],
                popxl.float32,
                name="in_stream_0",
            )
            a = ops.host_load(input0, "a")
            input1 = popxl.h2d_stream(
                [out_channel, int(in_channel / group), d_kernel, h_kernel, w_kernel],
                popxl.float32,
                name="in_stream_1",
            )
            b = ops.host_load(input1, "b")
            # custom conv.
            o = ops.conv(a, b, strides, padding, dilation, group)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: t, input1: weight})
        # conv in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_weight = torch.nn.Parameter(torch.tensor(weight).type(torch.float32))
        torch_outputs = nn.Conv3d(
            in_channel,
            out_channel,
            (d_kernel, h_kernel, w_kernel),
            strides,
            groups=group,
            padding=(1, 1, 1),
            dilation=dilation,
            bias=False,
        )
        torch_outputs.weight = torch_weight
        torch_output_data = torch_outputs(torch_t)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(
            torch_output_data.detach().numpy(),
            list(outputs.values())[0],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_conv2d_options(self):
        batch_size = 2
        in_channel = 10
        out_channel = 20
        height = 50
        width = 50
        h_kernel = 3
        w_kernel = 3
        strides = (1, 1)
        group = 1
        t = np.random.rand(batch_size, in_channel, height, width).astype("float16")
        weight = np.random.rand(
            out_channel, int(in_channel / group), h_kernel, w_kernel
        ).astype("float16")
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream(
                [batch_size, in_channel, height, width],
                popxl.float16,
                name="in_stream_0",
            )
            a = ops.host_load(input0, "a")
            input1 = popxl.h2d_stream(
                [out_channel, int(in_channel / group), h_kernel, w_kernel],
                popxl.float16,
                name="in_stream_1",
            )
            b = ops.host_load(input1, "b")
            # custom conv.
            o = ops.conv(
                a,
                b,
                strides,
                available_memory_proportions=[0.9],
                partials_types=["half"],
                enable_conv_dithering=[1],
            )
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: t, input1: weight})
        # conv in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_weight = torch.nn.Parameter(torch.tensor(weight).type(torch.float32))
        torch_outputs = nn.Conv2d(
            in_channel,
            out_channel,
            (h_kernel, w_kernel),
            tuple(strides),
            groups=group,
            bias=False,
        )
        torch_outputs.weight = torch_weight
        torch_output_data = torch_outputs(torch_t)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(
            torch_output_data.detach().numpy(),
            list(outputs.values())[0],
            rtol=1e-3,
            atol=1e-3,
        )
