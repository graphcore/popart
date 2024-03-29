# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Union
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor

from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def scaled_add(
    X: Tensor, Y: Tensor, a: Union[float, Tensor] = 1.0, b: Union[float, Tensor] = 1.0
) -> Tensor:
    """
    Perform a scaled addition of two tensors.

    Compute the sum of `X` scaled by `a and `Y` by `b`, which means `aX + bY`.

    Does not apply NumPy broadcasting.
    Uses mixed precision poplibs operations.

    `X` and `Y` must be the same shape, but can be different types.

    `a` and `b` must be scalars.

    Args:
        X, Y (Tensor): The tensor to be added.
        a, b (Union[float, Tensor]):
            Scalar that will be multiplied by `X` and `Y` before performing the addition.
    Returns:
        Tensor: A tensor containing `aX + bY`.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    tensors_to_check = dict(X=X, Y=Y)

    ins = {0: X.id, 1: Y.id}

    if isinstance(a, Tensor):
        tensors_to_check["a"] = a
        ins[2] = a.id
        a = 1.0

    if isinstance(b, Tensor):
        tensors_to_check["b"] = b
        ins[3] = b.id
        b = 1.0

    check_in_graph(g, **tensors_to_check)
    check_tensor_ipu_and_tile_set(**tensors_to_check)

    settings = ctx._get_op_settings("scaled_add")
    opid = _ir.OperatorIdentifier(
        "ai.graphcore", "ScaledAdd", 1, _ir.NumInputs(2, 4), 1
    )
    op = pb_g.createConnectedOp_ScaledAddOp(
        ins,
        {
            0: g._create_tensor_id("scaled_add_out"),
        },
        opid,
        a,
        b,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def scaled_add_(
    X: Tensor, Y: Tensor, a: Union[float, Tensor] = 1.0, b: Union[float, Tensor] = 1.0
) -> Tensor:
    """
    Perform a scaled addition of two tensors (in-place).

    Compute the sum of `X` scaled by `a and `Y` by `b`. This is performed in place on `X`, which means that `X = aX + bY`.

    Does not apply NumPy broadcasting.
    Uses mixed precision poplibs operations.

    `X` and `Y` must be the same shape, but can be different types.

    Args:
        X, Y: The tensor to be added.
        a, b:
            Scalar that will be multiplied by `X` and `Y` before performing the addition.
    Returns:
        Tensor:
            The `X` tensor containing `aX + bY`.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    tensors_to_check = dict(X=X, Y=Y)

    ins = {0: X.id, 1: Y.id}

    if isinstance(a, Tensor):
        tensors_to_check["a"] = a
        ins[2] = a.id
        a = 1.0

    if isinstance(b, Tensor):
        tensors_to_check["b"] = b
        ins[3] = b.id
        b = 1.0

    check_in_graph(g, **tensors_to_check)
    check_tensor_ipu_and_tile_set(**tensors_to_check)

    settings = ctx._get_op_settings("scaled_add")
    op = pb_g.createConnectedOp_ScaledAddLhsInplaceOp(
        ins,
        {
            0: g._create_tensor_id("scaled_add__" + X.name),
        },
        a,
        b,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
