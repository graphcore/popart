# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def detach(t: Tensor) -> Tensor:
    """
    Prevents gradient computation of this tensor.

    This operations is numerically equivlent to the identity op.

    Args:
        t: Tensor
            Input tensor.
    Returns:
        out: Tensor
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings('detach')
    opid = _ir.OperatorIdentifier("ai.graphcore", "Detach", 1,
                                  _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_DetachOp(
        {0: t.id},
        {0: g._create_tensor_id("detach_out")},
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def detach_(t: Tensor) -> Tensor:
    """
    Prevents gradient computation of this tensor (inplace).

    The inplace version of :func:`~ops.detach`. Behaviour is the same, but
    blocks gradient propagation inplace on the input tensor.

    Args:
        t: Tensor
            Input tensor.
    Returns:
        out: Tensor
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings('detach_inplace')
    op = pb_g.createConnectedOp_DetachInplaceOp(
        {0: t.id},
        {0: g._create_tensor_id("detach_out")},
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
