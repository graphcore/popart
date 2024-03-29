# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops

import popart._internal.ir as _ir

from utils import contains_op_of_type


class TestHostLoad:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            h2d = popxl.h2d_stream((), popxl.dtypes.float32)
            _ = ops.host_load(h2d, "x")

        assert len(g.tensors) == 3
        assert len(g.variables) == 0
        assert contains_op_of_type("HostLoad", _ir.op.exchange.HostLoadOp, g)
        assert contains_op_of_type("Init", _ir.op.InitOp, g)

    def test_within_graph(self):
        ir = popxl.Ir()
        g = ir.main_graph

        sg = ir.create_empty_graph("subgraph")

        with g:
            with sg:
                h2d = popxl.h2d_stream((), popxl.dtypes.float32)
                _ = ops.host_load(h2d, "x")

        assert len(g.tensors) == 1
        assert len(sg.tensors) == 2
        assert len(sg.variables) == 0
        assert contains_op_of_type("HostLoad", _ir.op.exchange.HostLoadOp, sg)
        assert contains_op_of_type("Init", _ir.op.InitOp, sg)
