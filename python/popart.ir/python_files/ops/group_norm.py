# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional
import popart._internal.ir as _ir
from popart.ir.context import get_current_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph

__all__ = ['group_norm', 'layer_norm']


def group_norm(x: Tensor,
               weight: Tensor,
               bias: Tensor,
               num_groups: int,
               eps: float = 1e-5) -> Tensor:
    """
    Applies Group Normalisation over a Tensor. https://arxiv.org/abs/1803.08494

    Args:
        x: Tensor
            Tensor to be normalized.
        weight: Tensor
            Tensor to scale output of normalisation.
        bias: Tensor
            Tensor to shift output of normalisation.
        num_groups:
            Number of groups to separate the channels into.
    Returns:
        out: Tensor
            The group normalised Tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, x, weight, bias)

    settings = ctx._get_op_settings('group_norm')
    opid = _ir.OperatorIdentifier("ai.graphcore", "GroupNormalization", 1,
                                  _ir.NumInputs(3, 3), 3)
    op = pb_g.createConnectedOp_GroupNormOp(
        {
            0: x.id,
            1: weight.id,
            2: bias.id
        },
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


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor,
               eps: float = 1e-5) -> Tensor:
    """
    Applies Layer Normalisation over a Tensor.
    Uses `group_norm` under the hood.
    
    Args:
        x: Tensor
            Tensor to be normalized.
        weight: Tensor
            Tensor to scale output of normalisation.
        bias: Tensor
            Tensor to shift output of normalisation.
    Returns:
        out: Tensor
            The layer normalised Tensor.
    """
    return group_norm(x, weight=weight, bias=bias, num_groups=1, eps=eps)
