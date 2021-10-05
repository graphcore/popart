# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart.ir as pir
import popart.ir.ops as ops

import popart._internal.ir as _ir

from utils import contains_op_of_type


class TestHostLoad:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            h2d = pir.h2d_stream((), pir.dtypes.float32)
            x = ops.host_load(h2d, "x")

        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 0
        assert contains_op_of_type("HostLoad", _ir.op.exchange.HostLoadOp, g)
        assert contains_op_of_type("Init", _ir.op.InitOp, g)
