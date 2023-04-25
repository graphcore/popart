# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def clip(t: Tensor, min: float = float("-inf"), max: float = float("inf")) -> Tensor:
    """
    Clip all elements so they are within the range [min, max]. NaN values are propagated.

    Args:
        t (Tensor): Input Tensor.
        min (float): min output range
        max (float): max output range

    Returns:
        Tensor: clipped tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings("clip")
    opid = _ir.OperatorIdentifier("ai.onnx", "Clip", 6, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_ClipOp(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("clip_out"),
        },
        opid,
        min,
        max,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


# Alias
clamp = clip
