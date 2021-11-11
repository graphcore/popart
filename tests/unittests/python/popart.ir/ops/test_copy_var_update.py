# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops

from utils import contains_op_of_type


class TestAccumulate:
    def test_add(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1)
            b = pir.constant(2)
            c = ops.var_updates.copy_var_update_(a, b)
        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 1
        assert contains_op_of_type("CopyVarUpdate", _ir.op.CopyVarUpdateOp, g)
