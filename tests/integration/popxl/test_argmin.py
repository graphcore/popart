# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np
import torch


class TestArgMin:
    def test_argmin_1(self):
        t = np.random.rand(10, 20).astype("float32")
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream([10, 20], popxl.float32, name="in_stream_0")
            a = ops.host_load(input0, "a")
            # custom argmin.
            o = ops.argmin(a, dim=0, keepdim=False)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: t})
        # argmin in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_outputs = torch_t.argmin(dim=0, keepdim=False)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(
            torch_outputs.detach().numpy(), list(outputs.values())[0], rtol=0, atol=0
        )

    def test_argmin_2(self):
        t = np.random.rand(5, 10, 20).astype("float32")
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream([5, 10, 20], popxl.float32, name="in_stream_0")
            a = ops.host_load(input0, "a")
            # custom argmin.
            o = ops.argmin(a, dim=-1, keepdim=True)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: t})
        # argmin in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_outputs = torch_t.argmin(dim=-1, keepdim=True)
        # compare the result between PopXL and torch
        np.testing.assert_allclose(
            torch_outputs.detach().numpy(), list(outputs.values())[0], rtol=0, atol=0
        )
