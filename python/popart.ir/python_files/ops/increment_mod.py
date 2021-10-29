# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popart.ir.context import get_current_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph

__all__ = ['increment_mod', 'increment_mod_']


def increment_mod(t: Tensor, increment: float, modulus: float) -> Tensor:
    """
    Compute `(t + increment) % modulus`

    Args:
        t: Tensor
            Tensor to increment (modulo)
        increment: float
            How much to increment the input tensor by.
        increment: float
            The modulo operand.
    Returns:
        out: Tensor
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t)

    settings = ctx._get_op_settings('increment_mod')
    opid = _ir.OperatorIdentifier("ai.graphcore", "IncrementMod", 1,
                                  _ir.NumInputs(1, 1), 1)
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


def increment_mod_(t: Tensor, increment: float, modulus: float) -> Tensor:
    """
    Compute `(t + increment) % modulus` inplace on `t`

    Args:
        t: Tensor
            Tensor to increment (modulo)
        increment: float
            How much to increment the input tensor by.
        increment: float
            The modulo operand.
    Returns:
        out: Tensor (alias of input `t`)
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t)

    settings = ctx._get_op_settings('increment_mod_inplace')
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
