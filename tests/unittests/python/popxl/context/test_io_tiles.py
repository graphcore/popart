# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl


def test_virtual_graph_with():
    ir = popxl.Ir()
    g = ir.main_graph

    with g:
        x = popxl.variable(1)
        with popxl.io_tiles():
            _ = x + 1

    ops = g._pb_graph.getOps()
    assert len(ops) == 1
    op: _ir.Op = ops[0]
    assert isinstance(op, _ir.op.AddOp)
    assert op.getSettings().tileSet == _ir.TileSet.IO
