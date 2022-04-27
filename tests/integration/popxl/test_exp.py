# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np
import torch


class TestExp:
    def test_exp(self):
        t = np.random.rand(10, 20).astype('float32')
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream([10, 20],
                                      popxl.float32,
                                      name="in_stream_0")
            a = ops.host_load(input0, "a")
            o = ops.exp(a)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: t})
        # exp in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_outputs = torch_t.exp()
        # compare the result between PopXL and torch
        np.testing.assert_allclose(torch_outputs.detach().numpy(),
                                   list(outputs.values())[0],
                                   rtol=1e-6,
                                   atol=1e-6)

    def test_exp_(self):
        t = np.random.rand(5, 10, 20).astype('float32')
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream([5, 10, 20],
                                      popxl.float32,
                                      name="in_stream_0")
            a = ops.host_load(input0, "a")
            o = ops.exp_(a)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: t})
        # exp in torch
        torch_t = torch.tensor(t).type(torch.float32)
        torch_outputs = torch_t.exp()
        # compare the result between PopXL and torch
        np.testing.assert_allclose(torch_outputs.detach().numpy(),
                                   list(outputs.values())[0],
                                   rtol=1e-6,
                                   atol=1e-6)
