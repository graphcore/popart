# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popart.ir import dtypes
from popart.ir.context import get_current_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph, cast_if_needed

__all__ = ['logical_not']


def logical_not(x: Tensor) -> Tensor:
    """
    Computes element-wise the value of NOT x.
    Inputs will be cast to bool if needed.

    Args:
        lhs, rhs: Tensor
            Tensors to be compared.
    Returns:
        out: Tensor
            The value (lhs NOT rhs)
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, x)

    x = cast_if_needed(x, dtypes.bool)

    settings = ctx._get_op_settings('not')
    opid = _ir.OperatorIdentifier("ai.onnx", "Not", 1, _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_NotOp(
        {0: x.id},
        {0: g._create_tensor_id("not_out")},
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
