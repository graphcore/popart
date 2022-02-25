# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
import pytest

from utils import contains_op_of_type


class TestAccumulate:
    def test_add(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            a = pir.variable(1)
            b = pir.constant(2)
            c = ops.var_updates.accumulate_(a, b)
        assert len(g.tensors) == 3
        assert len(g.variables) == 1
        assert contains_op_of_type("Accumulate", _ir.op.AccumulateOp, g)
        op = g._pb_graph.getOps()[0]
        op.getAccumulationType() == _ir.AccumulationType.Add

    def test_dampened_add_float(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            a = pir.variable(1)
            b = pir.constant(2)
            c = ops.var_updates.accumulate_(a, b, 0.9)
        assert len(g.tensors) == 3
        assert len(g.variables) == 1
        assert contains_op_of_type("Accumulate", _ir.op.AccumulateOp, g)
        op = g._pb_graph.getOps()[0]
        op.getAccumulationType() == _ir.AccumulationType.DampenedAdd

    def test_dampened_add_tensor(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            a = pir.variable(1)
            b = pir.constant(2)
            factor = pir.variable(0.9)
            c = ops.var_updates.accumulate_(a, b, factor)
        assert contains_op_of_type("Accumulate", _ir.op.AccumulateOp, g)
        op = g._pb_graph.getOps()[0]
        op.getAccumulationType() == _ir.AccumulationType.DampenedAdd

    def test_dampened_add_square(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            a = pir.variable(1)
            b = pir.constant(2)
            c = ops.var_updates.accumulate_square_(a, b, 0.999)
        assert contains_op_of_type("Accumulate", _ir.op.AccumulateOp, g)
        op = g._pb_graph.getOps()[0]
        op.getAccumulationType() == _ir.AccumulationType.DampenedAddSquare

    def test_mean(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            a = pir.variable(1)
            b = pir.constant(2)
            step = pir.variable(0)
            c = ops.var_updates.accumulate_mean_(a, b, step)
        assert contains_op_of_type("Accumulate", _ir.op.AccumulateOp, g)
        op = g._pb_graph.getOps()[0]
        op.getAccumulationType() == _ir.AccumulationType.Mean

    @pytest.mark.parametrize("constant_", [True, False])
    def test_scale(self, constant_):
        ir = pir.Ir()
        g = ir.main_graph
        f = 0.125

        with g:
            a = pir.variable(1)
            if not constant_:
                f = pir.variable(f)
            c = ops.var_updates.accumulator_scale_(a, f)
        assert contains_op_of_type("AccumulatorScale",
                                   _ir.op.AccumulatorScaleOp, g)
        op = g._pb_graph.getOps()[0]
        assert op.getFactor().val() == (0.125 if constant_ else 0.0)

    def test_zero(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            a = pir.variable(1)
            c = ops.var_updates.accumulator_zero_(a)
        assert contains_op_of_type("AccumulatorScale",
                                   _ir.op.AccumulatorScaleOp, g)

    def test_sparse_accumulate(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            a = pir.variable([2, 2])
            b = pir.variable([2, 2])
            c = pir.constant([2])
            axis = 1
            d = ops.var_updates.sparse_accumulate_(a, b, c, axis)
        assert contains_op_of_type("PopartSparseAccumulate",
                                   _ir.op.SparseAccumulateOp, g)
        op = g._pb_graph.getOps()[0]
        assert op.getAxis() == axis

    @pytest.mark.parametrize("constant_", [True, False])
    def test_sparse_accumulate_2(self, constant_):
        ir = pir.Ir()
        g = ir.main_graph
        f = 0.125

        with g:
            a = pir.variable([2, 2])
            b = pir.variable([2, 2])
            c = pir.constant([2])
            W = pir.variable([2, 2])
            if not constant_:
                f = pir.variable(f)
            axis = 1
            d = ops.var_updates.sparse_accumulate_(a, b, c, axis, f, W)

        assert contains_op_of_type("PopartSparseAccumulate",
                                   _ir.op.SparseAccumulateOp, g)
        op = g._pb_graph.getOps()[0]
        assert op.getFactor().val() == (0.125 if constant_ else 0.0)
        assert op.getAxis() == axis
