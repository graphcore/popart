# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import (
    get_current_context,
    op_debug_context,
)
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def batch_norm_inference(
    t: Tensor,
    scale: Tensor,
    bias: Tensor,
    mean: Tensor,
    var: Tensor,
    epsilon: float = 1e-5,
    momentum: float = 0.9,
) -> Tensor:
    """
    Apply batch normalisation to a tensor in an inference setting.

    For more details, refer to the paper :arxiv:`Group Normalization <1803.08494>`.

    Args:
        t (Tensor): Tensor to be normalized.
        scale (Tensor): Tensor used to scale the result of normalisation.
        bias (Tensor): Tensor used to shift the result of normalisation.
        mean (Tensor): Mean estimate.
        var (Tensor): Variance estimate.
        epsilon (float): small quantity for avoidance of div-by-zero when variance is zero.
        momentum (float): coefficient for the exponential moving average (not used in inference).
    Returns:
        Tensor:
            The batch normalised tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t, scale=scale, b=bias, mean=mean, var=var)
    check_tensor_ipu_and_tile_set(t=t, scale=scale, b=bias, mean=mean, var=var)

    settings = ctx._get_op_settings("batch_norm")

    opid = _ir.OperatorIdentifier(
        "ai.onnx", "BatchNormalization", 9, _ir.NumInputs(5, 5), 1
    )
    op = pb_g.createConnectedOp_BatchNormOp(
        {0: t.id, 1: scale.id, 2: bias.id, 3: mean.id, 4: var.id},
        {
            0: g._create_tensor_id("batch_norm_out"),
        },
        opid,
        epsilon,
        momentum,
        1,
        False,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
