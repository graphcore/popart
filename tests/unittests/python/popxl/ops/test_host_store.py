# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops

import popart._internal.ir as _ir

from utils import contains_op_of_type


class TestHostStore:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1.0, name="a")
            b = ops.add(a, a)

            # Adds a dummy ActGrad tensor in the main graph as the stream handle
            d2h = popxl.d2h_stream((), popxl.dtypes.float32)
            ops.host_store(d2h, b)

        assert len(g.tensors) == 3
        assert len(g.variables) == 1
        assert contains_op_of_type("HostStore", _ir.op.exchange.HostStoreOp, g)
