# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def add(lhs: Tensor, rhs: Tensor) -> Tensor:
    """
    Add two tensors element-wise.

    Follows NumPy broadcasting rules. Arguments must have the same dtype.

    Args:
        lhs, rhs: Tensor
            Tensors to be added.
    Returns:
        add: Tensor
            The sum of the input tensors
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, lhs=lhs, rhs=rhs)
    check_tensor_ipu_and_tile_set(lhs=lhs, rhs=rhs)

    settings = ctx._get_op_settings('add')
    opid = _ir.OperatorIdentifier("ai.onnx", "Add", 6, _ir.NumInputs(2, 2), 1)
    op = pb_g.createConnectedOp_AddOp(
        {
            0: lhs.id,
            1: rhs.id
        },
        {
            0: g._create_tensor_id("add_out"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def add_(lhs: Tensor, rhs: Tensor) -> Tensor:
    """
    Add two tensors element-wise (in-place).

    Follows NumPy broadcasting rules. Arguments must have the same dtype.

    Note: There is no add_rhs_inplace_ op, please use add_lhs_inplace_(rhs, lhs) or rhs += lhs for
        the same functionality.

    Args:
        lhs, rhs: Tensor
            Tensors to be added.
    Returns:
        lhs: Tensor
            The lhs tensor with rhs added in place.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, lhs=lhs, rhs=rhs)
    check_tensor_ipu_and_tile_set(lhs=lhs, rhs=rhs)

    settings = ctx._get_op_settings('add_lhs_inplace')
    opid = _ir.OperatorIdentifier("ai.graphcore", "AddLhsInplace", 1,
                                  _ir.NumInputs(2, 2), 1)
    op = pb_g.createConnectedOp_AddLhsInplaceOp(
        {
            0: lhs.id,
            1: rhs.id
        },
        {
            0: g._create_tensor_id("add_lhs_inplace_out"),
        },
        opid,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
