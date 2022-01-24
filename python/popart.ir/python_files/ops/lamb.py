# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from popart.ir.context import get_current_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph


def lamb_square(t: Tensor) -> Tensor:
    """
    Squares each element before applying an add reduction.

    Used in the LAMB optimizer: https://arxiv.org/abs/1904.00962

    Args:
        t: Tensor
            Input tensor.
    Returns:
        out: Tensor
            Output tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings('lamb_square')
    op = pb_g.createConnectedOp_LambSquareOp(
        {0: t.id}, {0: g._create_tensor_id(f"lamb_square_out")}, settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
