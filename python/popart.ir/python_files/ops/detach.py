# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popart.ir import dtypes
from popart.ir.context import get_current_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph

__all__ = ['detach']


def detach(x: Tensor) -> Tensor:
    """
    Prevents gradient computation of this tensor.
    Numerically equivlent to the idenity op.

    Args:
        x: Tensor
            Input tensor.
    Returns:
        out: Tensor
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, x)

    settings = ctx._get_op_settings('detach')
    opid = _ir.OperatorIdentifier("ai.graphcore", "Detach", 1,
                                  _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_DetachOp(
        {0: x.id},
        {0: g._create_tensor_id("detach_out")},
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))