# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
import popxl
import popxl.ops as ops
import numpy as np


@pytest.mark.parametrize("shape", [[10, 20], [5, 10, 15]])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_cumsum(shape, dim):
    t = np.random.rand(*shape).astype("float32")
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        input0 = popxl.h2d_stream(shape, popxl.float32, name="in_stream_0")
        a = ops.host_load(input0, "a")
        o = ops.cumsum(a, dim=dim)
        o_d2h = popxl.d2h_stream(o.shape, o.dtype, name="out_stream")
        ops.host_store(o_d2h, o)

    with popxl.Session(ir, "ipu_model") as session:
        outputs = session.run({input0: t})

    # numpy output
    numpy_outputs = np.cumsum(t, axis=dim)

    np.testing.assert_allclose(
        numpy_outputs,
        list(outputs.values())[0],
        rtol=1e-6,
        atol=1e-6
    )
