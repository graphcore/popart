# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def ceil(t: Tensor) -> Tensor:
    """
    Compute the ceil of the elements of input tensor. NaN values are propagated.

    Args:
        t (Tensor):
            Input tensor.

    Returns:
        Tensor: ceil output
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings("ceil")
    opid = _ir.OperatorIdentifier("ai.onnx", "Ceil", 1, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_CeilOp(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("ceil_out"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
