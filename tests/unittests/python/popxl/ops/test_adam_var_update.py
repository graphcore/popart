# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops

from utils import contains_op_of_type


class TestAdamVarUpdate:
    def test_adam_var_update_const(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            b = popxl.variable(2)
            c = popxl.constant(3)
            d = popxl.constant(4)
            _ = ops.var_updates.adam_var_update(a, b, c, d, 0.5, 0.1)
        assert len(g.tensors) == 5
        assert len(g.variables) == 2
        assert contains_op_of_type("AdamVarUpdate", _ir.op.AdamVarUpdateOp, g)

    def test_adam_var_update_w_nonconst(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            b = popxl.variable(2)
            c = popxl.constant(3)
            d = popxl.constant(4)
            lr = popxl.variable(1e-3)
            _ = ops.var_updates.adam_var_update(a, b, c, d, lr)
        assert len(g.tensors) == 6
        assert len(g.variables) == 3
        assert contains_op_of_type("AdamVarUpdate", _ir.op.AdamVarUpdateOp, g)

    def test_adam_var_update_m_nonconst(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            b = popxl.variable(2)
            c = popxl.constant(3)
            d = popxl.constant(4)
            mwn = popxl.variable(1e-5)
            _ = ops.var_updates.adam_var_update(a, b, c, d, None, mwn)
        assert len(g.tensors) == 6
        assert len(g.variables) == 3
        assert contains_op_of_type("AdamVarUpdate", _ir.op.AdamVarUpdateOp, g)

    def test_adam_var_update_both_nonconst(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            b = popxl.variable(2)
            c = popxl.constant(3)
            d = popxl.constant(4)
            lr = popxl.variable(1e-3)
            mwn = popxl.variable(1e-5)
            _ = ops.var_updates.adam_var_update(a, b, c, d, lr, mwn)
        assert len(g.tensors) == 7
        assert len(g.variables) == 4
        assert contains_op_of_type("AdamVarUpdate", _ir.op.AdamVarUpdateOp, g)
