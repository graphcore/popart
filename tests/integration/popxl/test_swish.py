# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np


class TestSwish:
    def test_swish(self):
        t = np.random.rand(10, 20).astype("float32")
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream([10, 20], popxl.float32, name="in_stream_0")
            a = ops.host_load(input0, "a")
            o = ops.swish(a)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: t})
        # swish in numpy
        np_outputs = 1 / (1 + np.exp(-t)) * t
        # compare the result between PopXL and numpy
        np.testing.assert_allclose(
            np_outputs,
            outputs[o_d2h],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_swish_(self):
        t = np.random.rand(5, 10, 20).astype("float32")
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # host load
            input0 = popxl.h2d_stream([5, 10, 20], popxl.float32, name="in_stream_0")
            a = ops.host_load(input0, "a")
            o = ops.swish_(a)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run({input0: t})
        # swish in numpy
        np_outputs = 1 / (1 + np.exp(-t)) * t
        # compare the result between PopXL and numpy
        np.testing.assert_allclose(
            np_outputs,
            outputs[o_d2h],
            rtol=1e-6,
            atol=1e-6,
        )
