# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np
import torch
from torch.nn import functional as F
import onnx.backend.test.case.node.resize as onnx_resize


class TestInterpolate:
    def test_interpolate_nearest(self):
        batch_size = 2
        in_channel = 10
        height = 10
        width = 10
        scale_factor = (1.0, 1.0, 2.0, 3.0)
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
            # custom interpolate.
            o = ops.interpolate(a, scale_factor, 'nearest')
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: t})
        # interpolate in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_outputs = F.interpolate(torch_t,
                                      scale_factor=(2.0, 3.0),
                                      mode='nearest',
                                      recompute_scale_factor=False)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(torch_outputs.detach().numpy(),
                                   list(outputs.values())[0],
                                   rtol=1e-7,
                                   atol=1e-7)

    def test_interpolate_linear(self):
        in_channel = 10
        height = 10
        width = 10
        scale_factor = (1.0, 1.0, 0.3)
        t = np.random.rand(in_channel, height, width).astype('float32')
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream([in_channel, height, width],
                                      popxl.float32,
                                      name="in_stream_0")
            a = ops.host_load(input0, "a")
            # custom interpolate.
            o = ops.interpolate(a, scale_factor, 'linear')
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: t})
        # interpolate in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_outputs = F.interpolate(torch_t,
                                      scale_factor=0.3,
                                      mode='linear',
                                      align_corners=False,
                                      recompute_scale_factor=False)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(torch_outputs.detach().numpy(),
                                   list(outputs.values())[0],
                                   rtol=1e-7,
                                   atol=1e-7)

    def test_interpolate_nearest_mode(self):
        batch_size = 1
        in_channel = 10
        height = 5
        width = 5
        scale_factor = (1.0, 1.0, 2.0, 2.0)
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
            # custom interpolate.
            o = ops.interpolate(a, scale_factor, 'nearest',
                                'round_prefer_ceil')
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: t})
        # interpolate in onnx
        out_onnx = onnx_resize.interpolate_nd(
            t,
            lambda x: onnx_resize.nearest_coeffs(x, mode='round_prefer_ceil'),
            scale_factors=scale_factor)
        # compare the result between PopXL and onnx
        np.testing.assert_allclose(out_onnx,
                                   list(outputs.values())[0],
                                   rtol=1e-7,
                                   atol=1e-7)

    def test_interpolate_coordinate_transformation_type(self):
        batch_size = 1
        in_channel = 10
        height = 5
        width = 5
        scale_factor = (1.0, 1.0, 2.0, 2.0)
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
            # custom interpolate.
            o = ops.interpolate(a,
                                scale_factor,
                                'linear',
                                coordinate_transformation_mode='align_corners')
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: t})
        # interpolate in onnx
        out_onnx = onnx_resize.interpolate_nd(
            t,
            onnx_resize.linear_coeffs,
            scale_factors=scale_factor,
            coordinate_transformation_mode='align_corners')
        # compare the result between PopXL and onnx
        np.testing.assert_allclose(out_onnx,
                                   list(outputs.values())[0],
                                   rtol=1e-7,
                                   atol=1e-7)
