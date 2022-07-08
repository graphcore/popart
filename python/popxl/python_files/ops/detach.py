# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def detach(t: Tensor) -> Tensor:
    """
    Prevent gradient computation of this tensor.

    This operation is numerically equivalent to the identity op.

    See also `PyTorch Tensor.detach <https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html>`__.

    Args:
        t (Tensor):
            Input tensor.
    Returns:
        Tensor: The input tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings("detach")
    opid = _ir.OperatorIdentifier("ai.graphcore", "Detach", 1, _ir.NumInputs(1, 1), 1)
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
    Prevent in-place gradient computation of this tensor.

    The in-place version of :func:`~popxl.ops.detach`. The behaviour is the same,
    it blocks gradient propagation on the input tensor but does not make
    a copy of the input tensor.

    See also `PyTorch Tensor.detach_ <https://pytorch.org/docs/stable/generated/torch.Tensor.detach_.html>`__.

    Args:
        t (Tensor):
            Input tensor.
    Returns:
        Tensor: The input tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings("detach_inplace")
    op = pb_g.createConnectedOp_DetachInplaceOp(
        {0: t.id},
        {0: g._create_tensor_id("detach_out")},
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
