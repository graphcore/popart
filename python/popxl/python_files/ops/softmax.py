# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, handle_negative_axis


@op_debug_context
def softmax(t: Tensor, axis: int) -> Tensor:
    """
    Computes the softmax on an input.

    This recales the slices of axis such that all elements are within [0, 1] and
    sum to 1. The output shape and dtype matches the input.

    Args:
        t: Tensor
            Tensor to be softmaxed.
        axis: int
            The axis along which the softmax will be computed.
    Returns:
        out: Tensor
            The softmaxed tensor
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings('softmax')
    opid = _ir.OperatorIdentifier("ai.onnx", "Softmax", 11, _ir.NumInputs(
        1, 1), 1)
    op = pb_g.createConnectedOp_SoftmaxOp(
        {0: t.id}, {0: g._create_tensor_id(f"softmax_out")}, opid,
        handle_negative_axis(t, axis), settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
