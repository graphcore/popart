# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops

from utils import contains_op_of_type


class TestAdamUpdater:
    def test_adam_updater_no_bias_no_wd(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            m = popxl.variable(1, name='m')
            v = popxl.variable(2, name='v')
            _ = ops.var_updates.adam_updater(m, v)

        assert len(g.tensors) == 3
        assert contains_op_of_type("AdamUpdater", _ir.op.AdamUpdaterOp, g)
        op = g._pb_graph.getOps()[0]
        assert op.isOptimizerOp()

    def test_adam_bias_updater(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            m = popxl.variable(1, name='m')
            v = popxl.variable(2, name='v')
            t = popxl.variable(2, name='t')
            b1 = 0.9
            b2 = 0.99
            _ = ops.var_updates.adam_updater(m,
                                             v,
                                             time_step=t,
                                             beta1=b1,
                                             beta2=b2)

        assert len(g.tensors) == 4
        assert contains_op_of_type("AdamUpdater", _ir.op.AdamUpdaterOp, g)

    def test_adam_updater_bias_invalid(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            m = popxl.variable(1, name='m')
            v = popxl.variable(2, name='v')
            t = popxl.variable(2, name='t')
            b1 = 0.9
            with pytest.raises(ValueError) as excinfo:
                _ = ops.var_updates.adam_updater(m, v, time_step=t, beta1=b1)
            message = str(excinfo.value)
        assert "Bias correction requires both beta1 and beta2 not None." in message

    def test_adam_wd_updater(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            w = popxl.variable(1, name='w')
            m = popxl.variable(1, name='m')
            v = popxl.variable(2, name='v')
            wd = popxl.constant(0.2, name='wd')

            _ = ops.var_updates.adam_updater(m, v, weight=w, weight_decay=wd)
        assert len(g.tensors) == 5
        assert contains_op_of_type("AdamUpdater", _ir.op.AdamUpdaterOp, g)

    def test_adam_wd_updater_invalid(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            m = popxl.variable(1, name='m')
            v = popxl.variable(2, name='v')
            t = popxl.variable(1, name='t')
            wd = popxl.constant(0.2, name='wd')
            with pytest.raises(ValueError) as excinfo:
                _ = ops.var_updates.adam_updater(m,
                                                 v,
                                                 time_step=t,
                                                 weight_decay=wd)
            message = str(excinfo.value)
        assert "Weight decay requires weight to be not None." in message

    def test_adam_bias_wd_updater(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            m = popxl.variable(1, name='m')
            v = popxl.variable(2, name='v')
            w = popxl.variable(1, name='w')
            t = popxl.variable(2, name='t')
            wd = popxl.constant(0.2, name='wd')
            b1 = 0.9
            b2 = 0.99
            _ = ops.var_updates.adam_updater(m, v, w, t, wd, b1, b2)

        assert len(g.tensors) == 6
        assert contains_op_of_type("AdamUpdater", _ir.op.AdamUpdaterOp, g)

    def test_lamb_updater_no_bias_no_wd(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            m = popxl.variable(1, name='m')
            v = popxl.variable(2, name='v')
            _ = ops.var_updates.lamb_updater(m, v)

        assert len(g.tensors) == 3
        assert contains_op_of_type("AdamUpdater", _ir.op.AdamUpdaterOp, g)
        op = g._pb_graph.getOps()[0]
        assert op.isOptimizerOp()

    def test_lamb_bias_updater(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            m = popxl.variable(1, name='m')
            v = popxl.variable(2, name='v')
            t = popxl.variable(2, name='t')
            b1 = 0.9
            b2 = 0.99
            _ = ops.var_updates.lamb_updater(m,
                                             v,
                                             time_step=t,
                                             beta1=b1,
                                             beta2=b2)

        assert len(g.tensors) == 4
        assert contains_op_of_type("AdamUpdater", _ir.op.AdamUpdaterOp, g)

    def test_lamb_updater_bias_invalid(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            m = popxl.variable(1, name='m')
            v = popxl.variable(2, name='v')
            t = popxl.variable(2, name='t')
            b1 = 0.9
            with pytest.raises(ValueError) as excinfo:
                _ = ops.var_updates.lamb_updater(m, v, time_step=t, beta1=b1)
            message = str(excinfo.value)
        assert "Bias correction requires both beta1 and beta2 not None." in message

    def test_lamb_wd_updater(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            w = popxl.variable(1, name='w')
            m = popxl.variable(1, name='m')
            v = popxl.variable(2, name='v')
            wd = popxl.constant(0.2, name='wd')

            _ = ops.var_updates.lamb_updater(m, v, weight=w, weight_decay=wd)
        assert len(g.tensors) == 5
        assert contains_op_of_type("AdamUpdater", _ir.op.AdamUpdaterOp, g)

    def test_lamb_wd_updater_invalid(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            m = popxl.variable(1, name='m')
            v = popxl.variable(2, name='v')
            t = popxl.variable(1, name='t')
            wd = popxl.constant(0.2, name='wd')
            with pytest.raises(ValueError) as excinfo:
                _ = ops.var_updates.lamb_updater(m,
                                                 v,
                                                 time_step=t,
                                                 weight_decay=wd)
            message = str(excinfo.value)
        assert "Weight decay requires weight to be not None." in message

    def test_lamb_bias_wd_updater(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            m = popxl.variable(1, name='m')
            v = popxl.variable(2, name='v')
            w = popxl.variable(1, name='w')
            t = popxl.variable(2, name='t')
            wd = popxl.constant(0.2, name='wd')
            b1 = 0.9
            b2 = 0.99
            _ = ops.var_updates.lamb_updater(m, v, w, t, wd, b1, b2)

        assert len(g.tensors) == 6
        assert contains_op_of_type("AdamUpdater", _ir.op.AdamUpdaterOp, g)

    def test_adamax_updater(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            m = popxl.variable(1, name='m')
            v = popxl.variable(2, name='v')
            t = popxl.variable(1, name='t')

            _ = ops.var_updates.adamax_updater(m, v, time_step=t)
        assert len(g.tensors) == 4
        assert contains_op_of_type("AdamUpdater", _ir.op.AdamUpdaterOp, g)

    def test_adamax_updater_invalid(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            m = popxl.variable(1, name='m')
            v = popxl.variable(2, name='v')
            with pytest.raises(ValueError) as excinfo:
                _ = ops.var_updates.adamax_updater(m, v)
            message = str(excinfo.value)
            assert "AdaMax requires time_step not None." in message

    def test_adamax_wd_updater(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            w = popxl.variable(1, name='w')
            m = popxl.variable(1, name='m')
            v = popxl.variable(2, name='v')
            t = popxl.variable(1, name='t')
            wd = popxl.constant(0.2, name='wd')

            _ = ops.var_updates.adamax_updater(m,
                                               v,
                                               weight=w,
                                               time_step=t,
                                               weight_decay=wd)
        assert len(g.tensors) == 6
        assert contains_op_of_type("AdamUpdater", _ir.op.AdamUpdaterOp, g)
