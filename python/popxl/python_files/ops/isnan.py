# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def isnan(t: Tensor) -> Tensor:
    """
    Return a boolean tensor of the same shape indicating which elements are NaN.

    Args:
        t (Tensor):
            Tensor to check.

    Returns:
        Tensor: boolean tensor of the same shape indicating which elements are NaN.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings("isnan")
    opid = _ir.OperatorIdentifier("ai.onnx", "IsNaN", 9, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_IsNaN(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("isnan_out"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
