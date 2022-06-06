# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
from popxl import ops


def test_remote_buffer_session():
    """Tests that Remote Buffers store and load correctly."""
    ir = popxl.Ir()
    with ir.main_graph, popxl.in_sequence():
        rb = popxl.RemoteBuffer((), popxl.float32, 1)

        ops.remote_store(rb, 0, popxl.constant(1, popxl.float32))
        y = ops.remote_load(rb, 0)

        d2h = popxl.d2h_stream(y.shape, y.dtype)
        ops.host_store(d2h, y)

    with popxl.Session(ir, "ipu_hw") as sess:
        assert sess.run()[d2h] == 1
