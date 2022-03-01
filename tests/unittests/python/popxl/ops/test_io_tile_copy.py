# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops


class TestIpuCopy:
    def test_copy_to(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            h2d = popxl.h2d_stream((), popxl.dtypes.float32)
            x = ops.host_load(h2d, "x")
            x_io = ops.io_tile_copy(x)

        g_ops = g._pb_graph.getOps()
        assert len(g_ops) == 3
        io_copy = g_ops[-1]
        assert isinstance(io_copy, _ir.op.IoTileCopyOp)

    def test_copy_from(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            with popxl.io_tiles():
                h2d = popxl.h2d_stream((), popxl.dtypes.float32)
                x = ops.host_load(h2d, "x")
            x_io = ops.io_tile_copy(x)

        g_ops = g._pb_graph.getOps()
        assert len(g_ops) == 3
        io_copy = g_ops[-1]
        assert isinstance(io_copy, _ir.op.IoTileCopyOp)
