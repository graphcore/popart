# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def increment_mod(t: Tensor, increment: float, modulus: float) -> Tensor:
    """
    Increment the elements of a tensor using modulo arithmetic.

    Args:
        t (Tensor):
            Tensor to increment (modulo).
        increment (float):
            How much to increment the input tensor by.
        modulus (float):
            The modulo operand.

    Returns:
        Tensor: A new tensor with result of `(t + increment) % modulus`.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings("increment_mod")
    opid = _ir.OperatorIdentifier(
        "ai.graphcore", "IncrementMod", 1, _ir.NumInputs(1, 1), 1
    )
    op = pb_g.createConnectedOp_IncrementModOp(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("increment_mod_out"),
        },
        opid,
        increment,
        modulus,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def increment_mod_(t: Tensor, increment: float, modulus: float) -> Tensor:
    """
    Increment the elements of a tensor using modulo arithmetic in place.

    Args:
        t (Tensor): Tensor to increment (modulo).
        increment (float): How much to increment the input tensor by.
        modulus (float): The modulo operand.

    Returns:
        Tensor: Alias of the input tensor with result of `(t + increment) % modulus`.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings("increment_mod_inplace")
    op = pb_g.createConnectedOp_IncrementModInplaceOp(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("increment_mod_inplace_out"),
        },
        increment,
        modulus,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
