# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def cos(t: Tensor) -> Tensor:
    """
    Compute the cosine of each element of the input tensor.

    See also `PyTorch Tensor.cos <https://pytorch.org/docs/stable/generated/torch.cos.html>`__.

    Args:
        t (Tensor): Input tensor
    Returns:
        Tensor: Output tensor
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, input=t)
    check_tensor_ipu_and_tile_set(input=t)

    settings = ctx._get_op_settings('cos')
    opid = _ir.OperatorIdentifier("ai.onnx", "Cos", 7, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_CosOp(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("cos_out"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
