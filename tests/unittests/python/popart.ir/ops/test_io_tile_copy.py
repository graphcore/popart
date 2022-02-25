# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops


class TestIpuCopy:
    def test_copy_to(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            h2d = pir.h2d_stream((), pir.dtypes.float32)
            x = ops.host_load(h2d, "x")
            x_io = ops.io_tile_copy(x)

        g_ops = g._pb_graph.getOps()
        assert len(g_ops) == 3
        io_copy = g_ops[-1]
        assert isinstance(io_copy, _ir.op.IoTileCopyOp)
        io_copy.getSettings().tileSet == _ir.TileSet.IO

    def test_copy_from(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            with pir.io_tiles():
                h2d = pir.h2d_stream((), pir.dtypes.float32)
                x = ops.host_load(h2d, "x")
            x_io = ops.io_tile_copy(x)

        g_ops = g._pb_graph.getOps()
        assert len(g_ops) == 3
        io_copy = g_ops[-1]
        assert isinstance(io_copy, _ir.op.IoTileCopyOp)
        io_copy.getSettings().tileSet == _ir.TileSet.Compute
