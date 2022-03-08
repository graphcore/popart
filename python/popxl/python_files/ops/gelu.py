# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def gelu(t: Tensor) -> Tensor:
    """
    Computes the GELU activation on a Tensor.

    See: https://arxiv.org/abs/1606.08415

    Args:
        t: Tensor
            Input tensor.
    Returns:
        out: Tensor
            Output tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings('gelu')
    opid = _ir.OperatorIdentifier("ai.graphcore", "Gelu", 1, _ir.NumInputs(
        1, 1), 1)
    op = pb_g.createConnectedOp_GeluOp(
        {0: t.id}, {0: g._create_tensor_id("gelu_out")}, opid, settings)

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def gelu_(t: Tensor) -> Tensor:
    """
    Computes the Gelu activation on a Tensor inplace.
    https://arxiv.org/abs/1606.08415

    Args:
        t: Tensor
            Input tensor.
    Returns:
        out: Tensor
            t with the gelu activation applied on it inplace.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings('gelu_inplace')
    op = pb_g.createConnectedOp_GeluInplaceOp(
        {0: t.id}, {0: g._create_tensor_id("gelu_inplace_out")}, settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
