# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popart.ir import dtypes
from popart.ir.context import get_current_context, op_debug_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set, cast_if_needed

__all__ = ['where']


@op_debug_context
def where(condition: Tensor, lhs: Tensor, rhs: Tensor) -> Tensor:
    """
    Element wise `lhs if condition else rhs`.
    Chooses either elements `lhs` or `rhs` depending on `condition`.
    The operator supports multi-directional broadcasting (numpy-style).

    Args:
        condition: Tensor
            A boolean Tensor where True indicates lhs element and false rhs element.
            The tensor will be casted to a bool if needed.
        lhs, rhs: Tensor
            Tensors choose elements from.
    Returns:
        out: Tensor
            Element wise `lhs if condition else rhs`.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, condition=condition, lhs=lhs, rhs=rhs)
    check_tensor_ipu_and_tile_set(condition=condition, lhs=lhs, rhs=rhs)

    condition = cast_if_needed(condition, dtypes.bool)

    settings = ctx._get_op_settings('where')
    opid = _ir.OperatorIdentifier("ai.onnx", "Where", 9, _ir.NumInputs(3, 3),
                                  1)
    op = pb_g.createConnectedOp_WhereOp(
        {
            0: condition.id,
            1: lhs.id,
            2: rhs.id,
        },
        {
            0: g._create_tensor_id("where_out"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
