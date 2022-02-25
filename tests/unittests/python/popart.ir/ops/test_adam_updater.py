# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops

from utils import contains_op_of_type


class TestAdamUpdater:
    def test_adam_updater_no_bias_no_wd(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            m = pir.variable(1, name='m')
            v = pir.variable(2, name='v')
            updater = ops.var_updates.adam_updater(m, v)

        assert len(g.tensors) == 3
        assert contains_op_of_type("AdamUpdater", _ir.op.AdamUpdaterOp, g)
        op = g._pb_graph.getOps()[0]
        assert op.isOptimizerOp()

    def test_adam_bias_updater(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            m = pir.variable(1, name='m')
            v = pir.variable(2, name='v')
            t = pir.variable(2, name='t')
            b1 = 0.9
            b2 = 0.99
            updater = ops.var_updates.adam_updater(m,
                                                   v,
                                                   time_step=t,
                                                   beta1=b1,
                                                   beta2=b2)

        assert len(g.tensors) == 4
        assert contains_op_of_type("AdamUpdater", _ir.op.AdamUpdaterOp, g)

    def test_adam_updater_bias_invalid(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            m = pir.variable(1, name='m')
            v = pir.variable(2, name='v')
            t = pir.variable(2, name='t')
            b1 = 0.9
            with pytest.raises(ValueError) as excinfo:
                updater = ops.var_updates.adam_updater(m,
                                                       v,
                                                       time_step=t,
                                                       beta1=b1)
            message = str(excinfo.value)
        assert "Bias correction requires both beta1 and beta2 not None." in message

    def test_adam_wd_updater(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            w = pir.variable(1, name='w')
            m = pir.variable(1, name='m')
            v = pir.variable(2, name='v')
            wd = pir.constant(0.2, name='wd')

            updater = ops.var_updates.adam_updater(m,
                                                   v,
                                                   weight=w,
                                                   weight_decay=wd)
        assert len(g.tensors) == 5
        assert contains_op_of_type("AdamUpdater", _ir.op.AdamUpdaterOp, g)

    def test_adam_wd_updater_invalid(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            m = pir.variable(1, name='m')
            v = pir.variable(2, name='v')
            t = pir.variable(1, name='t')
            wd = pir.constant(0.2, name='wd')
            with pytest.raises(ValueError) as excinfo:
                updater = ops.var_updates.adam_updater(m,
                                                       v,
                                                       time_step=t,
                                                       weight_decay=wd)
            message = str(excinfo.value)
        assert "Weight decay requires weight to be not None." in message

    def test_adam_bias_wd_updater(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            m = pir.variable(1, name='m')
            v = pir.variable(2, name='v')
            w = pir.variable(1, name='w')
            t = pir.variable(2, name='t')
            wd = pir.constant(0.2, name='wd')
            b1 = 0.9
            b2 = 0.99
            updater = ops.var_updates.adam_updater(m, v, w, t, wd, b1, b2)

        assert len(g.tensors) == 6
        assert contains_op_of_type("AdamUpdater", _ir.op.AdamUpdaterOp, g)

    def test_lamb_updater_no_bias_no_wd(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            m = pir.variable(1, name='m')
            v = pir.variable(2, name='v')
            updater = ops.var_updates.lamb_updater(m, v)

        assert len(g.tensors) == 3
        assert contains_op_of_type("AdamUpdater", _ir.op.AdamUpdaterOp, g)
        op = g._pb_graph.getOps()[0]
        assert op.isOptimizerOp()

    def test_lamb_bias_updater(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            m = pir.variable(1, name='m')
            v = pir.variable(2, name='v')
            t = pir.variable(2, name='t')
            b1 = 0.9
            b2 = 0.99
            updater = ops.var_updates.lamb_updater(m,
                                                   v,
                                                   time_step=t,
                                                   beta1=b1,
                                                   beta2=b2)

        assert len(g.tensors) == 4
        assert contains_op_of_type("AdamUpdater", _ir.op.AdamUpdaterOp, g)

    def test_lamb_updater_bias_invalid(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            m = pir.variable(1, name='m')
            v = pir.variable(2, name='v')
            t = pir.variable(2, name='t')
            b1 = 0.9
            with pytest.raises(ValueError) as excinfo:
                updater = ops.var_updates.lamb_updater(m,
                                                       v,
                                                       time_step=t,
                                                       beta1=b1)
            message = str(excinfo.value)
        assert "Bias correction requires both beta1 and beta2 not None." in message

    def test_lamb_wd_updater(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            w = pir.variable(1, name='w')
            m = pir.variable(1, name='m')
            v = pir.variable(2, name='v')
            wd = pir.constant(0.2, name='wd')

            updater = ops.var_updates.lamb_updater(m,
                                                   v,
                                                   weight=w,
                                                   weight_decay=wd)
        assert len(g.tensors) == 5
        assert contains_op_of_type("AdamUpdater", _ir.op.AdamUpdaterOp, g)

    def test_lamb_wd_updater_invalid(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            m = pir.variable(1, name='m')
            v = pir.variable(2, name='v')
            t = pir.variable(1, name='t')
            wd = pir.constant(0.2, name='wd')
            with pytest.raises(ValueError) as excinfo:
                updater = ops.var_updates.lamb_updater(m,
                                                       v,
                                                       time_step=t,
                                                       weight_decay=wd)
            message = str(excinfo.value)
        assert "Weight decay requires weight to be not None." in message

    def test_lamb_bias_wd_updater(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            m = pir.variable(1, name='m')
            v = pir.variable(2, name='v')
            w = pir.variable(1, name='w')
            t = pir.variable(2, name='t')
            wd = pir.constant(0.2, name='wd')
            b1 = 0.9
            b2 = 0.99
            updater = ops.var_updates.lamb_updater(m, v, w, t, wd, b1, b2)

        assert len(g.tensors) == 6
        assert contains_op_of_type("AdamUpdater", _ir.op.AdamUpdaterOp, g)

    def test_adamax_updater(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            m = pir.variable(1, name='m')
            v = pir.variable(2, name='v')
            t = pir.variable(1, name='t')

            updater = ops.var_updates.adamax_updater(m, v, time_step=t)
        assert len(g.tensors) == 4
        assert contains_op_of_type("AdamUpdater", _ir.op.AdamUpdaterOp, g)

    def test_adamax_updater_invalid(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            m = pir.variable(1, name='m')
            v = pir.variable(2, name='v')
            with pytest.raises(ValueError) as excinfo:
                updater = ops.var_updates.adamax_updater(m, v)
            message = str(excinfo.value)
            assert "AdaMax requires time_step not None." in message

    def test_adamax_wd_updater(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            w = pir.variable(1, name='w')
            m = pir.variable(1, name='m')
            v = pir.variable(2, name='v')
            t = pir.variable(1, name='t')
            wd = pir.constant(0.2, name='wd')

            updater = ops.var_updates.adamax_updater(m,
                                                     v,
                                                     weight=w,
                                                     time_step=t,
                                                     weight_decay=wd)
        assert len(g.tensors) == 6
        assert contains_op_of_type("AdamUpdater", _ir.op.AdamUpdaterOp, g)
