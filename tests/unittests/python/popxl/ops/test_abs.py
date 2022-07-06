# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops

from utils import contains_op_of_type


class TestAbs:
    def test_abs(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(-1)
            _ = ops.abs(a)
        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        assert contains_op_of_type("Abs", _ir.op.AbsOp, g)
