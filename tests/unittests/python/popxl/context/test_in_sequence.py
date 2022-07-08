# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import List, Type
import numpy as np
import pytest

import popart._internal.ir as _ir
import popxl
from popxl.context import _execution_context


def test_in_sequence_with():
    ir = popxl.Ir()
    g = ir.main_graph

    with g:
        small = popxl.variable(1, name="small")
        big = popxl.variable(np.ones((2, 2), np.float32), name="big")

        # From a liveness perspective,
        #   these two Ops are in the wrong order.
        with popxl.in_sequence():
            _ = big + 1
            _ = small * 1

    ops = g._pb_graph.getOpSchedule()
    assert isinstance(ops[0], _ir.op.AddOp)
    assert isinstance(ops[1], _ir.op.MulOp)


def test_in_sequence_fn():
    ir = popxl.Ir()
    g = ir.main_graph

    @popxl.in_sequence()
    def fn(small, big):
        b = big + 1
        a = small * 1
        return a, b

    with g:
        small = popxl.variable(1, name="small")
        big = popxl.variable(np.ones((2, 2), np.float32), name="big")
        fn(small, big)

    ops = g._pb_graph.getOpSchedule()
    assert isinstance(ops[0], _ir.op.AddOp)
    assert isinstance(ops[1], _ir.op.MulOp)


def is_sequence_of_ops(ops: List[_ir.Op], op_types: List[Type[_ir.Op]]):
    if len(ops) != len(op_types):
        raise ValueError("List of Ops does not match the list of Op types")

    for op, op_type in zip(ops, op_types):
        if not isinstance(op, op_type):
            return False

    return True


def test_nested_in_sequence():
    ir = popxl.Ir()
    g = ir.main_graph

    @popxl.in_sequence(False)
    def badly_written_layer(small, big):
        # These two should swap when not in_sequence
        b = big + 1
        a = small * 1
        return a, b

    with g:
        small = popxl.variable(1, name="small")
        big = popxl.variable(np.ones((2, 2), np.float32), name="big")
        with popxl.in_sequence(True):
            _ = big - 1
            badly_written_layer(small, big)
            _ = small / 1

    ops = g._pb_graph.getOpSchedule()
    assert is_sequence_of_ops(
        ops, [_ir.op.SubtractOp, _ir.op.MulOp, _ir.op.AddOp, _ir.op.DivOp]
    )


def test_none_not_allowed():
    with pytest.raises(TypeError) as excinfo:
        with popxl.in_sequence(None):
            pass
    assert "`None` cannot be passed to" in str(excinfo.value)


def test_in_sequence_false_first():
    ir = popxl.Ir()
    g = ir.main_graph

    with g:
        small = popxl.variable(1, name="small")
        big = popxl.variable(np.ones((2, 2), np.float32), name="big")

        # From a liveness perspective,
        #   these two Ops are in the wrong order.
        with popxl.in_sequence(False):
            _ = big + 1
            _ = small * 1

    ops = g._pb_graph.getOpSchedule()
    assert isinstance(ops[0], _ir.op.MulOp)
    assert isinstance(ops[1], _ir.op.AddOp)


def test_in_sequence_execution_context():
    ir = popxl.Ir()
    g = ir.main_graph

    with g:
        x = popxl.variable(np.ones((2, 2), np.float32), name="big")

        with popxl.in_sequence(True):
            _ = x + 1
            with _execution_context(_ir.ExecutionContext.WeightsFromHostFragment):
                _ = x - 1

    # If a topocon is created between (x+1) and (x-1) then a scheduling error will occur
    g._pb_graph.getOpSchedule()


def test_in_sequence_pass():
    """test `pass` mode.
    4 ops are created but only ops 1, 3, and 4 (not 2) have topological constraints applied"""
    ir = popxl.Ir()
    g = ir.main_graph

    with g:
        x = popxl.variable(np.ones((2, 2), np.float32), name="big")

        with popxl.in_sequence(True):
            x = x - 1

            with popxl.in_sequence("pass"):
                x = x * 1

            x = x + 1
            x = x / 1

    ops = g._pb_graph.getOpSchedule()
    assert is_sequence_of_ops(
        ops, [_ir.op.SubtractOp, _ir.op.MulOp, _ir.op.AddOp, _ir.op.DivOp]
    ) or is_sequence_of_ops(
        ops, [_ir.op.SubtractOp, _ir.op.AddOp, _ir.op.DivOp, _ir.op.MulOp]
    )
