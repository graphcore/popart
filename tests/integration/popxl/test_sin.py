# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np
import torch


class TestSin:
    def test_sin(self):
        input_data = np.random.rand(10, 20).astype('float32')
        ir = popxl.Ir()
        main = ir.main_graph

        with main:
            t = popxl.constant(input_data, popxl.float32, name="input_0")
            o = ops.sin(t)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)

        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run()

        # sin in torch
        torch_t = torch.tensor(input_data).type(torch.float32)
        torch_outputs = torch_t.sin()

        # compare the result between PopXL and torch
        np.testing.assert_allclose(torch_outputs.detach().numpy(),
                                   list(outputs.values())[0],
                                   rtol=1e-6,
                                   atol=1e-6)
