# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops
from utils import contains_op_of_type


class TestGreater:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable([1, 2, 4])
            b = popxl.variable([0, 2, 6])
            _ = ops.greater(a, b)
        assert len(g.tensors) == 3
        assert len(g.variables) == 2
        assert contains_op_of_type("Greater", _ir.op.GreaterOp, g)
