# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
import numpy as np
from popxl import dtypes


class TestShapedDropout:
    def test_shaped_dropout(self):
        np.random.seed(10)
        t = np.random.rand(8, 2).astype("float32")
        seed_tensors = popxl.create_seeds(10)
        ratio = 0.5
        shape = [8, 1]
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            # create seed tensor
            seed_h2d = popxl.h2d_stream(
                shape=(2,), dtype=dtypes.uint32, name="seed_stream"
            )
            seed = ops.host_load(seed_h2d, "seed")
            # host load
            input0 = popxl.h2d_stream([8, 2], popxl.float32, name="in_stream_0")
            a = ops.host_load(input0, "a")
            # shaped dropout.
            o = ops.shaped_dropout(a, seed_tensor=seed, shape=shape, ratio=ratio)
            # host store
            o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
            ops.host_store(o_d2h, o)
        # get the result
        with popxl.Session(ir, "ipu_model") as session:
            output = session.run({input0: t, seed_h2d: seed_tensors})
        # reference result
        reference_output = t / (1.0 - ratio)
        reference_output[list(output.values())[0] == 0] = 0
        # compare the result between PopXL and reference
        np.testing.assert_allclose(
            reference_output, list(output.values())[0], rtol=0, atol=0
        )
