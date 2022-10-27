# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from typing_extensions import Literal

CommGroup = _ir.CommGroup
CommGroupType = _ir.CommGroupType

OP_MAP = {
    "add": _ir.CollectiveOperator.Add,
    "mean": _ir.CollectiveOperator.Mean,
    "mul": _ir.CollectiveOperator.Mean,
    "min": _ir.CollectiveOperator.Min,
    "max": _ir.CollectiveOperator.Max,
    "and": _ir.CollectiveOperator.LogicalAnd,
    "or": _ir.CollectiveOperator.LogicalOr,
    "square_add": _ir.CollectiveOperator.SquareAdd,
}

CollectiveOps = Literal["add", "mean", "mul", "min", "max", "and", "or", "square_add"]


def to_collective_op(op: CollectiveOps) -> _ir.CollectiveOperator:
    try:
        return OP_MAP[op]
    except KeyError:
        raise ValueError(
            f"Not a valid op: {op}. Must choose from: {', '.join(OP_MAP.keys())}"
        )


def _rearrange_input(t, axis):
    """Rearange input so axis is first axis and flatten array to prepare for collective."""
    # Rearrange tensor
    if axis > 0:
        # Permute the concat axis to the front
        permutation = list(range(len(t.shape)))
        permutation.pop(axis)
        permutation.insert(0, axis)

        t = t.transpose(permutation)

    preshape = list(t.shape)

    # Collectives implicitly flatten tensors but need this
    # so autodiff creates correct grad op
    t = t.flatten()
    return t, preshape


def _rearrange_output(y, new_shape, axis):
    """Reshape collective output to correct shape and rearrange tensor so that axis order are correct."""
    # Reshape as collectives implicitly flattens
    y = y.reshape(new_shape)

    # Rearrange tensor back
    if axis > 0:
        # Permute the concat axis back to position
        permutation = list(range(len(y.shape)))
        permutation.pop(0)
        permutation.insert(axis, 0)

        y = y.transpose(permutation)
    return y
