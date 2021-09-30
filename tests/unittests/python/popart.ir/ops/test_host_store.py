# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart.ir as pir
import popart.ir.ops as ops

import popart._internal.ir as _ir

from utils import contains_op_of_type


class TestHostStore:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1, name="a")
            b = ops.add(a, a)

            # Adds a dummy ActGrad tensor in the main graph as the stream handle
            d2h = pir.d2h_stream(pir.dtypes.float32, ())
            ops.host_store(d2h, b)

        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 1
        contains_op_of_type("HostStore", _ir.op.exchange.HostStoreOp, g)
