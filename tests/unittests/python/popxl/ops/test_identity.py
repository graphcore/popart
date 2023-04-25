# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl
import popxl.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestIdentity:
    def test_fn_identity(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones((1, 2, 3)))
            _ = ops.identity(a)
        assert len(g.tensors) == 2
        assert contains_op_of_type("Identity", _ir.op.IdentityOp, g)

    def test_fn_rename(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(np.ones((1, 2, 3)))
            b = ops.identity(a, output_name="out123")
        assert b.name == "out123"
        assert len(g.tensors) == 2
        assert contains_op_of_type("Identity", _ir.op.IdentityOp, g)
