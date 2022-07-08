# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import (
    debug_context_frame_offset,
    get_current_context,
    op_debug_context,
)
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def group_norm(
    t: Tensor, weight: Tensor, bias: Tensor, num_groups: int, eps: float = 1e-5
) -> Tensor:
    """
    Apply group normalisation to a tensor.

    For more details, refer to the paper :arxiv:`Group Normalization <1803.08494>`.

    Args:
        t (Tensor): Tensor to be normalized.
        weight (Tensor): Tensor used to scale the result of normalisation.
        bias (Tensor): Tensor used to shift the result of normalisation.
        num_groups (int): Number of groups to separate the channels into.
        eps (float): The small value to use to avoid division by zero.
    Returns:
        Tensor:
            The group normalised tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t, weight=weight, bias=bias)
    check_tensor_ipu_and_tile_set(t=t, weight=weight, bias=bias)

    settings = ctx._get_op_settings("group_norm")
    opid = _ir.OperatorIdentifier(
        "ai.graphcore", "GroupNormalization", 1, _ir.NumInputs(3, 3), 3
    )
    op = pb_g.createConnectedOp_GroupNormOp(
        {0: t.id, 1: weight.id, 2: bias.id},
        {
            0: g._create_tensor_id("group_norm_out"),
            1: g._create_tensor_id("group_norm_mean"),
            2: g._create_tensor_id("group_norm_inv_std_dev"),
        },
        opid,
        num_groups,
        eps,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


@debug_context_frame_offset(1)
def layer_norm(t: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    """
    Apply layer normalisation to a tensor.

    Uses `group_norm` under the hood.

    Args:
        t (Tensor): The tensor to be normalized.
        weight (Tensor): Tensor used to scale the result of normalisation.
        bias (Tensor): Tensor used to shift result of normalisation.
        eps (float): The small value to use to avoid division by zero
    Returns:
        Tensor: The layer normalised tensor.
    """
    return group_norm(t, weight=weight, bias=bias, num_groups=1, eps=eps)
