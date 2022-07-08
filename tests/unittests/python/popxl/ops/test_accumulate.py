# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops
import pytest

from utils import contains_op_of_type


class TestAccumulate:
    def test_add(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            b = popxl.constant(2)
            _ = ops.var_updates.accumulate_(a, b)
        assert len(g.tensors) == 3
        assert len(g.variables) == 1
        assert contains_op_of_type("Accumulate", _ir.op.AccumulateOp, g)
        _ = g._pb_graph.getOps()[0]

    def test_dampened_add_float(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            b = popxl.constant(2)
            _ = ops.var_updates.accumulate_(a, b, 0.9)
        assert len(g.tensors) == 3
        assert len(g.variables) == 1
        assert contains_op_of_type("Accumulate", _ir.op.AccumulateOp, g)
        _ = g._pb_graph.getOps()[0]

    def test_dampened_add_tensor(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            b = popxl.constant(2)
            factor = popxl.variable(0.9)
            _ = ops.var_updates.accumulate_(a, b, factor)
        assert contains_op_of_type("Accumulate", _ir.op.AccumulateOp, g)
        _ = g._pb_graph.getOps()[0]

    def test_dampened_add_square(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            b = popxl.constant(2)
            _ = ops.var_updates.accumulate_square_(a, b, 0.999)
        assert contains_op_of_type("Accumulate", _ir.op.AccumulateOp, g)
        _ = g._pb_graph.getOps()[0]

    def test_mean(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            b = popxl.constant(2)
            step = popxl.variable(0)
            _ = ops.var_updates.accumulate_mean_(a, b, step)
        assert contains_op_of_type("Accumulate", _ir.op.AccumulateOp, g)
        _ = g._pb_graph.getOps()[0]

    @pytest.mark.parametrize("constant_", [True, False])
    def test_scale(self, constant_):
        ir = popxl.Ir()
        g = ir.main_graph
        f = 0.125

        with g:
            a = popxl.variable(1)
            if not constant_:
                f = popxl.variable(f)
            _ = ops.var_updates.accumulator_scale_(a, f)
        assert contains_op_of_type("AccumulatorScale", _ir.op.AccumulatorScaleOp, g)
        op = g._pb_graph.getOps()[0]
        assert op.getFactor().val() == (0.125 if constant_ else 0.0)

    def test_zero(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            _ = ops.var_updates.accumulator_zero_(a)
        assert contains_op_of_type("AccumulatorScale", _ir.op.AccumulatorScaleOp, g)

    def test_sparse_accumulate(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable([2, 2])
            b = popxl.variable([2, 2])
            c = popxl.constant([2])
            axis = 1
            _ = ops.var_updates.sparse_accumulate_(a, b, c, axis)
        assert contains_op_of_type(
            "PopartSparseAccumulate", _ir.op.SparseAccumulateOp, g
        )
        op = g._pb_graph.getOps()[0]
        assert op.getAxis() == axis

    @pytest.mark.parametrize("constant_", [True, False])
    def test_sparse_accumulate_2(self, constant_):
        ir = popxl.Ir()
        g = ir.main_graph
        f = 0.125

        with g:
            a = popxl.variable([2, 2])
            b = popxl.variable([2, 2])
            c = popxl.constant([2])
            W = popxl.variable([2, 2])
            if not constant_:
                f = popxl.variable(f)
            axis = 1
            _ = ops.var_updates.sparse_accumulate_(a, b, c, axis, f, W)

        assert contains_op_of_type(
            "PopartSparseAccumulate", _ir.op.SparseAccumulateOp, g
        )
        op = g._pb_graph.getOps()[0]
        assert op.getFactor().val() == (0.125 if constant_ else 0.0)
        assert op.getAxis() == axis
