# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def div(lhs: Tensor, rhs: Tensor) -> Tensor:
    """
    Divide two tensors element-wise.

    Follows NumPy broadcasting rules. Arguments must have the same dtype.
    Output will be the same dtype as the inputs.
    With integer values, floor division is used.

    Args:
        lhs: Tensor
            Divisor
        rhs: Tensor
            Dividend
    Returns:
        mul: Tensor
            The division of lhs by rhs
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, lhs=lhs, rhs=rhs)
    check_tensor_ipu_and_tile_set(lhs=lhs, rhs=rhs)

    settings = ctx._get_op_settings('div')
    opid = _ir.OperatorIdentifier("ai.onnx", "Div", 7, _ir.NumInputs(2, 2), 1)
    op = pb_g.createConnectedOp_DivOp(
        {
            0: lhs.id,
            1: rhs.id
        },
        {
            0: g._create_tensor_id("div_out"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
