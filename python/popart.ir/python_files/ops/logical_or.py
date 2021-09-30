# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popart.ir import dtypes
from popart.ir.context import get_current_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph, cast_if_needed

__all__ = ['logical_or']


def logical_or(lhs: Tensor, rhs: Tensor) -> Tensor:
    """
    Compares two Tensors element-wise with an OR operator.
    Follows numpy broadcasting rules.
    Inputs will be cast to bool if needed.

    Args:
        lhs, rhs: Tensor
            Tensors to be compared.
    Returns:
        out: Tensor
            The value (lhs OR rhs)
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, lhs, rhs)

    lhs = cast_if_needed(lhs, dtypes.bool)
    rhs = cast_if_needed(rhs, dtypes.bool)

    settings = ctx._get_op_settings('or')
    opid = _ir.OperatorIdentifier("ai.onnx", "Or", 7, _ir.NumInputs(2, 2), 1)
    op = pb_g.createConnectedOp_OrOp(
        {
            0: lhs.id,
            1: rhs.id
        },
        {
            0: g._create_tensor_id("or_out"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
