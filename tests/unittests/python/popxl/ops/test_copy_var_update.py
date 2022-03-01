# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops

from utils import contains_op_of_type


class TestAccumulate:
    def test_add(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            b = popxl.constant(2)
            c = ops.var_updates.copy_var_update_(a, b)
        assert len(g.tensors) == 3
        assert len(g.variables) == 1
        assert contains_op_of_type("CopyVarUpdate", _ir.op.CopyVarUpdateOp, g)
