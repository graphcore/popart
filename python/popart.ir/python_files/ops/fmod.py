# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popart.ir.context import get_current_context, op_debug_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def fmod(lhs: Tensor, rhs: Tensor) -> Tensor:
    """
    Modulo two Tensors element-wise.

    Follows numpy broadcasting rules. Arguments must have the same dtype.

    Args:
        lhs, rhs: Tensor
            Tensors to be modded.
    Returns:
        fmod: Tensor
            The fmod of lhs and rhs
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, lhs=lhs, rhs=rhs)
    check_tensor_ipu_and_tile_set(lhs=lhs, rhs=rhs)

    settings = ctx._get_op_settings('fmod')
    opid = _ir.OperatorIdentifier("ai.graphcore", "Fmod", 1, _ir.NumInputs(
        2, 2), 1)
    op = pb_g.createConnectedOp_FmodOp(
        {
            0: lhs.id,
            1: rhs.id
        },
        {
            0: g._create_tensor_id("fmod_out"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
