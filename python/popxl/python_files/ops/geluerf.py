# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def geluerf(t: Tensor) -> Tensor:
    """
    Compute the accurate GELU activation on a tensor.

    For more details, refer to the paper :arxiv:`Gaussian Error Linear Units <1606.08415>`

    Args:
        t (Tensor):
            The input tensor.
    Returns:
        Tensor:
            A new tensor with the accurate GELU activation applied to it.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings("geluerf")
    opid = _ir.OperatorIdentifier("ai.graphcore", "GeluErf", 1, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_GeluErfOp(
        {0: t.id}, {0: g._create_tensor_id("geluerf_out")}, opid, settings
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def geluerf_(t: Tensor) -> Tensor:
    """
    Compute the accurate GELU activation on a tensor (in-place).

    For more details, refer to the paper :arxiv:`Gaussian Error Linear Units <1606.08415>`

    Args:
        t (Tensor):
            The input tensor.
    Returns:
        Tensor:
            The input tensor with the accurate GELU activation applied to it.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings("geluerf_inplace")
    op = pb_g.createConnectedOp_GeluErfInplaceOp(
        {0: t.id}, {0: g._create_tensor_id("geluerf_inplace_out")}, settings
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
