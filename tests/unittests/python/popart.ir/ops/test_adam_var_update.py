# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops

from utils import contains_op_of_type


class TestAdamVarUpdate:
    def test_adam_var_update_const(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1)
            b = pir.variable(2)
            c = pir.constant(3)
            d = pir.constant(4)
            e = ops.var_updates.adam_var_update(a, b, c, d, 0.5, 0.1)
        assert len(g.get_tensors()) == 5
        assert len(g.get_variables()) == 2
        assert contains_op_of_type("AdamVarUpdate", _ir.op.AdamVarUpdateOp, g)

    def test_adam_var_update_w_nonconst(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1)
            b = pir.variable(2)
            c = pir.constant(3)
            d = pir.constant(4)
            lr = pir.variable(1e-3)
            f = ops.var_updates.adam_var_update(a, b, c, d, lr)
        assert len(g.get_tensors()) == 6
        assert len(g.get_variables()) == 3
        assert contains_op_of_type("AdamVarUpdate", _ir.op.AdamVarUpdateOp, g)

    def test_adam_var_update_m_nonconst(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1)
            b = pir.variable(2)
            c = pir.constant(3)
            d = pir.constant(4)
            mwn = pir.variable(1e-5)
            f = ops.var_updates.adam_var_update(a, b, c, d, None, mwn)
        assert len(g.get_tensors()) == 6
        assert len(g.get_variables()) == 3
        assert contains_op_of_type("AdamVarUpdate", _ir.op.AdamVarUpdateOp, g)

    def test_adam_var_update_both_nonconst(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1)
            b = pir.variable(2)
            c = pir.constant(3)
            d = pir.constant(4)
            lr = pir.variable(1e-3)
            mwn = pir.variable(1e-5)
            f = ops.var_updates.adam_var_update(a, b, c, d, lr, mwn)
        assert len(g.get_tensors()) == 7
        assert len(g.get_variables()) == 4
        assert contains_op_of_type("AdamVarUpdate", _ir.op.AdamVarUpdateOp, g)
