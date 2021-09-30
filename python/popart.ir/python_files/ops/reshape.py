# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, Tuple
import numpy as np
import popart._internal.ir as _ir
from popart.ir.context import get_current_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph

__all__ = ["reshape"]


def handle_negative_axis(t: Tensor, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    replacement = t.nelms // np.prod(np.abs(shape))
    return tuple(axis if axis > 0 else replacement for axis in shape)


def reshape(t: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """
    Reshape a Tensor.

    Args:
        t: Tensor
            Tensor to be reshaped.
        shape: tuple of ints
            Tuple containing the shape of the output.
    Returns:
        out: Tensor
            The reshaped tensor
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t)

    if any(map(lambda axis: axis < -1, shape)) or shape.count(0) > 0:
        raise ValueError(
            f"Invalid shape value in reshape. Must be '>0' or '-1'. Provided {shape}."
        )

    if shape.count(-1) > 1:
        raise ValueError(
            f"Reshape shape can contain at most one '-1' value. Provided {shape}."
        )

    settings = ctx._get_op_settings('reshape')
    opid = _ir.OperatorIdentifier("ai.onnx", "Reshape", 5, _ir.NumInputs(1, 1),
                                  1)
    op = pb_g.createConnectedOp_ReshapeOp(
        {0: t.id}, {0: g._create_tensor_id(f"{t.name}_reshaped")}, opid,
        handle_negative_axis(t, shape), settings, False)

    return Tensor._from_pb_tensor(op.outTensor(0))
