# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popart.ir import dtypes
from popart.ir.globals import gcg
from popart.ir.tensor import Tensor
from .utils import check_in_graph, cast_if_needed

__all__ = ['logical_and']


def logical_and(lhs: Tensor, rhs: Tensor) -> Tensor:
    """
    Compares two Tensors element-wise with an AND operator.
    Follows numpy broadcasting rules.
    Inputs will be cast to bool if needed.

    Args:
        lhs, rhs: Tensor
            Tensors to be compared.
    Returns:
        out: Tensor
            The value (lhs AND rhs)
    """
    g = gcg()
    pb_g = g._pb_graph

    check_in_graph(g, lhs, rhs)

    lhs = cast_if_needed(lhs, dtypes.bool)
    rhs = cast_if_needed(rhs, dtypes.bool)

    settings = _ir.Settings(pb_g, 'and')
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
