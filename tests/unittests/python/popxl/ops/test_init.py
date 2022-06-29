# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops

import popart._internal.ir as _ir

from utils import contains_op_of_type


class TestInit:
    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            _ = ops.init((), popxl.dtypes.float32)

        assert len(g.tensors) == 1
        assert len(g.variables) == 0
        assert contains_op_of_type("Init", _ir.op.InitOp, g)
