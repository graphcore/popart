# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl import dtypes
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, cast_if_needed, check_tensor_ipu_and_tile_set


@op_debug_context
def logical_and(lhs: Tensor, rhs: Tensor) -> Tensor:
    """
    Apply a logical AND element-wise.

    Follows NumPy broadcasting rules.
    Inputs will be cast to bool if needed.

    Args:
        lhs, rhs: Tensor
            Tensors to be compared.
    Returns:
        out: Tensor
            The value (lhs AND rhs)
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, lhs=lhs, rhs=rhs)
    check_tensor_ipu_and_tile_set(lhs=lhs, rhs=rhs)

    lhs = cast_if_needed(lhs, dtypes.bool)
    rhs = cast_if_needed(rhs, dtypes.bool)

    settings = ctx._get_op_settings('and')
    opid = _ir.OperatorIdentifier("ai.onnx", "And", 7, _ir.NumInputs(2, 2), 1)
    op = pb_g.createConnectedOp_AndOp(
        {
            0: lhs.id,
            1: rhs.id
        },
        {
            0: g._create_tensor_id("and_out"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
