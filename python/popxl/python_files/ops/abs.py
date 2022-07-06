# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def abs(t: Tensor) -> Tensor:
    """
    Compute the absolute value of each element of the input tensor.

    See also `PyTorch Tensor.abs <https://pytorch.org/docs/stable/generated/torch.abs.html>`__.

    Args:
        t (Tensor): Input tensor
    Returns:
        Tensor: Output tensor
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, x=t)
    check_tensor_ipu_and_tile_set(x=t)

    settings = ctx._get_op_settings('abs')
    opid = _ir.OperatorIdentifier("ai.onnx", "Abs", 6, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_AbsOp(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("abs_out"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
