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
            c = ops.accumulate(a, b)
        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 1
        assert contains_op_of_type("Accumulate", _ir.op.AccumulateOp, g)
        op = g._pb_graph.getOps()[0]
        op.getAccumulationType() == _ir.AccumulationType.Add

    def test_dampened_add_float(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1)
            b = pir.constant(2)
            c = ops.accumulate(a, b, 0.9)
        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 1
        assert contains_op_of_type("Accumulate", _ir.op.AccumulateOp, g)
        op = g._pb_graph.getOps()[0]
        op.getAccumulationType() == _ir.AccumulationType.DampenedAdd

    def test_dampened_add_tensor(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1)
            b = pir.constant(2)
            factor = pir.variable(0.9)
            c = ops.accumulate(a, b, factor)
        assert contains_op_of_type("Accumulate", _ir.op.AccumulateOp, g)
        op = g._pb_graph.getOps()[0]
        op.getAccumulationType() == _ir.AccumulationType.DampenedAdd

    def test_dampened_add_square(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1)
            b = pir.constant(2)
            c = ops.accumulate_square(a, b, 0.999)
        assert contains_op_of_type("Accumulate", _ir.op.AccumulateOp, g)
        op = g._pb_graph.getOps()[0]
        op.getAccumulationType() == _ir.AccumulationType.DampenedAddSquare

    def test_mean(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1)
            b = pir.constant(2)
            step = pir.variable(0)
            c = ops.accumulate_mean(a, b, step)
        assert contains_op_of_type("Accumulate", _ir.op.AccumulateOp, g)
        op = g._pb_graph.getOps()[0]
        op.getAccumulationType() == _ir.AccumulationType.Mean
