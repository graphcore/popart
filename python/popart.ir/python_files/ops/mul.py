# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popart.ir.context import get_current_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph

__all__ = ['mul']


def mul(lhs: Tensor, rhs: Tensor) -> Tensor:
    """
    Multiplies two Tensors element-wise.
    Follows numpy broadcasting rules.
    Arguments must have the same dtype.
    
    Args:
        lhs, rhs: Tensor
            Tensors to be multiplied.
    Returns:
        mul: Tensor
            The product of lhs and rhs
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, lhs, rhs)

    settings = ctx._get_op_settings('mul')
    opid = _ir.OperatorIdentifier("ai.onnx", "Mul", 7, _ir.NumInputs(2, 2), 1)

    op = pb_g.createConnectedOp_MulOp(
        {
            0: lhs.id,
            1: rhs.id
        },
        {
            0: g._create_tensor_id("mul_out"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
