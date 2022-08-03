# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def swish(t: Tensor) -> Tensor:
    """
    Compute the Swish activation of a tensor.

    For more details, refer to `Rectifier (neural networks) <https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`__.

    Args:
        t (Tensor):
            The input tensor to calculate the activation of.
    Returns:
        Tensor:
            A tensor containing the activations.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)
    check_tensor_ipu_and_tile_set(t=t)

    settings = ctx._get_op_settings("swish")
    opid = _ir.OperatorIdentifier("ai.graphcore", "Swish", 1, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_SwishOp(
        {0: t.id},
        {
            0: g._create_tensor_id("swish_out"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def swish_(t: Tensor) -> Tensor:
    """
    Compute the Swish activation of a tensor in place.

    For more details, refer to `Rectifier (neural networks) <https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`__.

    Args:
        t (Tensor):
            The input tensor to calculate the activation of.
    Returns:
        Tensor:
            The input tensor with the Swish activation applied to it.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings("swish_inplace")
    op = pb_g.createConnectedOp_SwishInplaceOp(
        {0: t.id}, 
        {
            0: g._create_tensor_id("swish_inplace_out")
        }, 
        settings
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
